"""Train backbones and save Q/Z."""

import argparse
from pathlib import Path
import time

import torch

from src.backbones import Backbone, build_adjacency, preprocess, train
from src.data import DATASETS, load_dataset


def context_name(sigma, draw):
    return f"s{sigma:g}_d{draw}.pt"


# Reuse noise draws across severities and backbones.
def generate(args):
    torch.set_num_threads(args.threads)
    datasets = DATASETS if "all" in args.datasets else args.datasets
    sigmas = sorted(set(args.sigmas) | {0.})
    contexts = [(0., 0)] + [(sigma, draw) for draw in args.draws for sigma in sigmas if sigma > 0]
    for dataset in datasets:
        data = load_dataset(dataset, args.data_root, splits=args.splits)
        x_raw, y, edges = data["x"], data["y"], data["edge_index"]
        mode = data.get("preprocessing", "identity")
        dataset_dir = args.output / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        graph_path = dataset_dir / "graph.pt"
        if not graph_path.exists():
            torch.save({"edge_index": edges, "y": y}, graph_path)
        for kind in args.backbones:
            adjacency = None if kind == "mlp" else build_adjacency(edges, len(y), kind, args.device)
            for split, indices in enumerate(data["splits"]):
                x = preprocess(x_raw, indices["train_index"], mode)
                scale = x[indices["train_index"]].std(0, correction=1)
                for seed in args.seeds:
                    unit = dataset_dir / kind / f"split_{split}" / f"seed_{seed}"
                    checkpoint_path = unit / "checkpoint.pt"
                    missing = [(sigma, draw) for sigma, draw in contexts
                               if not (unit / context_name(sigma, draw)).exists()]
                    if checkpoint_path.exists() and not missing:
                        continue
                    if not checkpoint_path.exists() and any(unit.glob("s*_d*.pt")):
                        raise FileNotFoundError(f"{unit} contains predictions but its frozen checkpoint is missing")
                    unit.mkdir(parents=True, exist_ok=True)
                    started = time.monotonic()
                    if checkpoint_path.exists():
                        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                        model = Backbone(checkpoint["in_features"], checkpoint["out_features"],
                                         checkpoint["recipe"], adjacency).to(args.device)
                        model.load_state_dict(checkpoint["state_dict"])
                        model.eval().requires_grad_(False)
                    else:
                        model, checkpoint = train(x, y, indices, kind, dataset, seed,
                                                  adjacency, args.device, args.epochs)
                        checkpoint.update(dataset=dataset, backbone=kind, split=split,
                                          preprocessing=mode, num_nodes=len(y))
                        torch.save(checkpoint, checkpoint_path)
                    torch.save(indices, unit / "split.pt")
                    dtype = torch.float32 if dataset == "ogbn-products" else torch.float64
                    metadata = dict(dataset=dataset, backbone=kind, split=split, seed=seed,
                                    preprocessing=mode, best_epoch=checkpoint["best_epoch"])
                    for draw in (0, *args.draws):
                        selected = [(sigma, d) for sigma, d in missing if d == draw]
                        if not selected:
                            continue
                        noise = None if draw == 0 else torch.randn(
                            x.shape, dtype=x.dtype, generator=torch.Generator().manual_seed(draw))
                        for sigma, draw in selected:
                            corrupted = x if sigma == 0 else x + sigma * scale * noise
                            with torch.no_grad():
                                z = model(corrupted.to(args.device)).to(device="cpu", dtype=dtype)
                                q = z.softmax(1)
                            torch.save({"Q": q, "Z": z,
                                        "metadata": metadata | {"sigma": sigma, "draw": draw}},
                                       unit / context_name(sigma, draw))
                        del noise
                    del model
                    print(f"{dataset}/{kind}/split_{split}/seed_{seed}: "
                          f"saved {len(missing)} contexts in {time.monotonic() - started:.1f}s", flush=True)
            del adjacency


def add_arguments(parser):
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("runs/paper"))
    parser.add_argument("--datasets", nargs="+", choices=("all", *DATASETS), default=["all"])
    parser.add_argument("--backbones", nargs="+", choices=["mlp", "gcn", "sage"], default=["mlp", "gcn", "sage"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--draws", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0., .5, 1., 1.5, 2.])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--epochs", type=int, help="Override the 500-epoch training limit for a short trial")
    return parser


def main():
    parser = add_arguments(argparse.ArgumentParser(description=__doc__))
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
