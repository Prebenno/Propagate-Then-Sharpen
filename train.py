"""
Train backbones or optionally reuse pretrained models and saved Q/Z
"""

import argparse
import shutil
from pathlib import Path
import time

import torch

from src.backbones import Backbone, build_adjacency, preprocess, train
from src.data import DATASETS, load_dataset, preflight_inputs
from src.pretrained import DEFAULT_ROOT, available_checkpoints, unit_directory
from src.seeds import CLEAN_DRAW, MODEL_SEEDS, NOISY_DRAWS, corruption_seed


def context_name(sigma, draw):
    return f"s{sigma:g}_d{draw}.pt"


def _save(value, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _copy(source, path):
    temporary = path.with_name(path.name + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(path)


def generate(args):
    torch.set_num_threads(args.threads)
    datasets = DATASETS if "all" in args.datasets else args.datasets
    pretrained_root = getattr(args, "pretrained", None)
    seeds = args.seeds if args.seeds is not None else list(MODEL_SEEDS)
    if pretrained_root is not None and args.seeds is None:
        seeds = sorted({int(path.parent.name[5:]) for path in available_checkpoints(pretrained_root)
                        if path.relative_to(pretrained_root).parts[0] in datasets
                        and path.relative_to(pretrained_root).parts[1] in args.backbones})
        if not seeds:
            raise FileNotFoundError("No pretrained models found for the requested datasets/backbones")
    if pretrained_root is not None:
        pretrained_root = Path(pretrained_root)
        requested = [unit_directory(pretrained_root, dataset, kind, split, seed) / "checkpoint.pt"
                     for dataset in datasets for kind in args.backbones
                     for split in range(1 if dataset.startswith("ogbn-") else args.splits)
                     for seed in seeds]
        requested += [pretrained_root / dataset / "splits" / f"split_{split}.pt"
                      for dataset in datasets
                      for split in range(1 if dataset.startswith("ogbn-") else args.splits)]
        missing_files = [str(path) for path in requested if not path.is_file()]
        if missing_files:
            raise FileNotFoundError("Missing pretrained files: " + ", ".join(missing_files))
    preflight_inputs(datasets, args.data_root)
    requested_sigmas = args.sigmas if args.sigmas is not None else ([0.] if pretrained_root is not None else [0., .5, 1., 1.5, 2.])
    sigmas = sorted(set(requested_sigmas) | {0.})
    if any(sigma < 0 for sigma in sigmas) or any(draw < 0 for draw in args.draws):
        raise ValueError("Noise levels and noisy draw IDs must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        draws = args.draws
        contexts = [(0., CLEAN_DRAW)] + [(sigma, draw) for draw in draws for sigma in sigmas if sigma > 0]
        data = load_dataset(dataset, args.data_root, splits=args.splits)
        x_raw, y, edges = data["x"], data["y"], data["edge_index"]
        dataset_dir = args.output / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        graph_path = dataset_dir / "graph.pt"
        if not graph_path.exists():
            _save({"edge_index": edges, "y": y}, graph_path)
        for kind in args.backbones:
            adjacency = None
            for split, default_indices in enumerate(data["splits"]):
                indices = default_indices
                if pretrained_root is not None:
                    indices = torch.load(pretrained_root / dataset / "splits" / f"split_{split}.pt",
                                         map_location="cpu", weights_only=True)
                for seed in seeds:
                    unit = unit_directory(args.output, dataset, kind, split, seed)
                    checkpoint_path = unit / "checkpoint.pt"
                    split_path = unit / "split.pt"
                    source = None if pretrained_root is None else unit_directory(
                        pretrained_root, dataset, kind, split, seed)
                    unit.mkdir(parents=True, exist_ok=True)
                    started = time.monotonic()
                    copied = 0
                    if source is not None:
                        if not checkpoint_path.exists():
                            _copy(source / "checkpoint.pt", checkpoint_path)
                        if not split_path.exists():
                            _save(indices, split_path)
                        for sigma, draw in contexts:
                            name = context_name(sigma, draw)
                            saved = source / name
                            if sigma == 0 and not saved.is_file():
                                saved = source / "s0_d0.pt"  
                            if not (unit / name).exists() and saved.is_file():
                                _copy(saved, unit / name)
                                copied += 1
                    missing = [(sigma, draw) for sigma, draw in contexts
                               if not (unit / context_name(sigma, draw)).exists()]
                    if checkpoint_path.exists() and not missing:
                        if copied:
                            print(f"{dataset}/{kind}/split_{split}/seed_{seed}: reused {copied} saved Q/Z files", flush=True)
                        continue
                    if adjacency is None and kind != "mlp":
                        adjacency = build_adjacency(edges, len(y), kind, args.device)
                    checkpoint = None
                    if checkpoint_path.exists():
                        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                    mode = (checkpoint or {}).get("preprocessing", data.get("preprocessing", "identity"))
                    x = preprocess(x_raw, indices["train_index"], mode)
                    scale = x[indices["train_index"]].std(0, correction=1)
                    if checkpoint is not None:
                        model = Backbone(checkpoint["in_features"], checkpoint["out_features"],
                                         checkpoint["recipe"], adjacency).to(args.device)
                        model.load_state_dict(checkpoint["state_dict"])
                        model.eval().requires_grad_(False)
                    else:
                        model, checkpoint = train(x, y, indices, kind, dataset, seed,
                                                  adjacency, args.device, args.epochs, split=split)
                        checkpoint.update(dataset=dataset, backbone=kind, split=split,
                                          preprocessing=mode, num_nodes=len(y))
                        _save(checkpoint, checkpoint_path)
                    _save(indices, split_path)
                    dtype = torch.float32 if dataset == "ogbn-products" else torch.float64
                    metadata = dict(dataset=dataset, backbone=kind, split=split, seed=seed,
                                    preprocessing=mode, best_epoch=checkpoint["best_epoch"])
                    for draw in dict.fromkeys((CLEAN_DRAW, *draws)):
                        selected = [(sigma, d) for sigma, d in missing if d == draw]
                        if not selected:
                            continue
                        field_seed = None if draw == CLEAN_DRAW else corruption_seed(dataset, split, draw)
                        noise = None if field_seed is None else torch.randn(
                            x.shape, dtype=x.dtype, generator=torch.Generator().manual_seed(field_seed))
                        for sigma, draw in selected:
                            corrupted = x if sigma == 0 else x + sigma * scale * noise
                            with torch.no_grad():
                                z = model(corrupted.to(args.device)).to(device="cpu", dtype=dtype)
                                q = z.softmax(1)
                            _save({"Q": q, "Z": z,
                                        "metadata": metadata | {"sigma": sigma, "draw": draw, "corruption_seed": field_seed}},
                                       unit / context_name(sigma, draw))
                        del noise
                    del model
                    print(f"{dataset}/{kind}/split_{split}/seed_{seed}: "
                          f"reused {copied}, saved {len(missing)} Q/Z files in {time.monotonic() - started:.1f}s", flush=True)
            del adjacency


def add_arguments(parser):
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("runs/paper"))
    parser.add_argument("--datasets", nargs="+", choices=("all", *DATASETS), default=["all"])
    parser.add_argument("--backbones", nargs="+", choices=["mlp", "gcn", "sage"], default=["mlp", "gcn", "sage"])
    parser.add_argument("--seeds", type=int, nargs="+", help="Model seeds (default: saved pretrained seed IDs; training 0 1 2)")
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--draws", type=int, nargs="+", default=list(NOISY_DRAWS), help="Noisy draw IDs, not a count (default: 0 1 2)")
    parser.add_argument("--sigmas", type=float, nargs="+", help="Noise levels (default: pretrained 0; training 0 .5 1 1.5 2)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--epochs", type=int, help="Override the 500-epoch training limit for a short trial")
    parser.add_argument("--pretrained", type=Path, nargs="?", const=DEFAULT_ROOT,
                        help="Reuse pretrained models, original splits and available Q/Z files")
    return parser


def main():
    parser = add_arguments(argparse.ArgumentParser(description=__doc__))
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
