"""Load graphs and prepare splits."""

import argparse
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

import numpy as np
import torch
from torch_geometric.datasets import HeterophilousGraphDataset, WikiCS

from .seeds import split_seed



MAIN = ("wikics", "cora-tag", "pubmed-tag", "tape-arxiv23", "ogbn-arxiv",
        "ogbn-products", "ele-photo", "ele-computers", "books-history")
CONTROLS = ("roman-empire", "amazon-ratings")
DATASETS = MAIN + CONTROLS
CUSTOM = ("cora-tag", "pubmed-tag", "tape-arxiv23", "ele-photo", "ele-computers", "books-history")
CSTAG = {"ele-photo": "Photo", "ele-computers": "Computers", "books-history": "History"}
FEATURES = {
    "cora-tag": "locally encoded frozen RoBERTa-base CLS, 512 tokens (768 dimensions)",
    "pubmed-tag": "locally encoded frozen RoBERTa-base CLS, 512 tokens (768 dimensions)",
    "tape-arxiv23": "the paper's 300-dimensional features (encoder not recorded)",
    "ele-photo": "CS-TAG Photo_roberta_base_512_cls.npy",
    "ele-computers": "CS-TAG Computers_roberta_base_512_cls.npy",
    "books-history": "CS-TAG History_roberta_base_512_cls.npy",
}


def _extract(value):
    from torch_geometric.data import Data

    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, Data):
        value = value.to_dict()
    return value


def _read(path):
    return _extract(torch.load(path, map_location="cpu", weights_only=False))


def _random_splits(n, count, dataset):
    train, val = round(.6 * n), round(.2 * n)
    result = []
    for split in range(count):
        order = np.random.default_rng(split_seed(dataset, split)).permutation(n)
        parts = (order[:train], order[train:train + val], order[train + val:])
        result.append({name: torch.from_numpy(np.sort(index)).long()
                       for name, index in zip(("train_index", "val_index", "test_index"), parts)})
    return result


def _mask_splits(data, count):
    masks = [torch.as_tensor(data[name]).bool()
             for name in ("train_mask", "val_mask", "test_mask")]
    return [{name: (mask[:, split] if mask.ndim == 2 else mask).nonzero().flatten().long()
             for name, mask in zip(("train_index", "val_index", "test_index"), masks)}
            for split in range(count)]


def _official(name, root, count):
    directory = root / name
    if name == "wikics":
        cached = directory / "processed/data_undirected.pt"
        data = _read(cached) if cached.exists() else _extract(WikiCS(str(directory), is_undirected=True)[0])
        return data, _mask_splits(data, count)
    if name in CONTROLS:
        cached = directory / name.replace("-", "_") / "processed/data.pt"
        data = _read(cached) if cached.exists() else _extract(HeterophilousGraphDataset(str(directory), name=name)[0])
        return data, _random_splits(len(data["y"]), count, name)
    ogb_directory = directory / name.replace("-", "_")
    cached = ogb_directory / "processed/geometric_data_processed.pt"
    if cached.exists():
        data = _read(cached)
        split_name = "time" if name == "ogbn-arxiv" else "sales_ranking"
        indices = {role: torch.from_numpy(np.loadtxt(ogb_directory / "split" / split_name / f"{role}.csv.gz",
                                                    delimiter=",", dtype=np.int64)).reshape(-1)
                   for role in ("train", "valid", "test")}
    else:
        if ogb_directory.exists() and not (ogb_directory / "RELEASE_v1.txt").exists():
            # An incomplete local archive should not trigger OGB's interactive
            # upgrade-and-delete prompt during an unattended paper run.
            raise FileNotFoundError(f"Incomplete OGB input at {ogb_directory}. "
                                    "Restore the processed tensor and split CSVs or use a fresh data root.")
        from unittest.mock import patch
        from ogb.nodeproppred import PygNodePropPredDataset
        from torch_geometric.data import Data
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
        from torch_geometric.data.storage import GlobalStorage
        with torch.serialization.safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]), \
                patch("ogb.nodeproppred.dataset_pyg.decide_download", return_value=True):
            dataset = PygNodePropPredDataset(name=name, root=str(directory))
        data, indices = _extract(dataset[0]), dataset.get_idx_split()
    return data, [{f"{role}_index": indices["valid" if role == "val" else role].long()
                   for role in ("train", "val", "test")}]


def load_dataset(name, root, splits=10):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    if splits < 1 or (name == "wikics" and splits > 20):
        raise ValueError("Splits must be positive; WikiCS provides 20 splits")
    root = Path(root)
    if name in CUSTOM:
        path = root / name / "graph.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Supply {FEATURES[name]} in a graph.pt with x, y and edge_index.")
        data = _read(path)
        indices = _random_splits(len(data["y"]), splits, name)
    else:
        data, indices = _official(name, root, splits)
    x = data["x"]
    x = (x.to_dense() if x.layout != torch.strided else x).float().cpu()
    y = data["y"].long().reshape(-1).cpu()
    from torch_geometric.utils import remove_self_loops, to_undirected
    edges = data["edge_index"].long().cpu()
    edges, _ = remove_self_loops(edges)
    edges = to_undirected(edges, num_nodes=len(x)).contiguous()
    return {"x": x, "y": y, "edge_index": edges, "splits": indices, "preprocessing": "identity"}



def required_files(name, root):
    """Processed inputs required for an offline run; no checksums or manifest."""
    directory = Path(root) / name
    if name in CUSTOM:
        return [directory / "graph.pt"]
    if name == "wikics":
        return [directory / "processed/data_undirected.pt"]
    if name in CONTROLS:
        return [directory / name.replace("-", "_") / "processed/data.pt"]
    directory = directory / name.replace("-", "_")
    split_name = "time" if name == "ogbn-arxiv" else "sales_ranking"
    return [directory / "processed/geometric_data_processed.pt", *[
        directory / "split" / split_name / f"{role}.csv.gz" for role in ("train", "valid", "test")]]


def preflight_inputs(names, root):
    """Check the entire requested cohort before any classifier is trained."""
    missing = [str(path) for name in names for path in required_files(name, root) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing dataset inputs (prepare all data before training):\n" + "\n".join(missing))


def _extract_archive(source, root):
    root = Path(root).resolve()
    with zipfile.ZipFile(source) as archive:
        members = []
        for entry in archive.infolist():
            if entry.filename == "README.txt":
                continue
            name = PurePosixPath(entry.filename)
            if (name.is_absolute() or ".." in name.parts or "\\" in entry.filename
                    or not name.parts or name.parts[0] != "data"
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError(f"Unexpected path in paper-data archive: {entry.filename}")
            relative = Path(*name.parts[1:])
            target = (root / relative).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Archive path escapes data root: {entry.filename}")
            members.append((entry, target))
        for entry, target in members:
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            with archive.open(entry) as reader, temporary.open("wb") as writer:
                shutil.copyfileobj(reader, writer)
            temporary.replace(target)


def prepare_datasets(names, root, source=None, splits=10):
    """Unpack released feature inputs and prepare official caches sequentially."""
    names = tuple(DATASETS if "all" in names else names)
    unknown = set(names) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    if source is not None:
        source = Path(source)
        if source.is_dir():
            candidates = [source / "data/graphs", source / "data", source]
            source_root = next((candidate for candidate in candidates if any(
                path.is_file() for name in names for path in required_files(name, candidate))), None)
            if source_root is None:
                raise FileNotFoundError(f"No processed dataset inputs found in {source}")
            for name in names:
                for original in required_files(name, source_root):
                    if not original.is_file():
                        continue
                    destination = Path(root) / original.relative_to(source_root)
                    if destination.resolve() == original.resolve():
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(destination.name + ".tmp")
                    shutil.copyfile(original, temporary)
                    temporary.replace(destination)
        else:
            _extract_archive(source, root)
    # Detect absent paper-only embeddings before attempting any downloads.
    missing = [str(path) for name in names if name in CUSTOM
               for path in required_files(name, root) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Supply paper-data.zip for these exact feature inputs:\n" + "\n".join(missing))
    for name in names:
        data = load_dataset(name, root, splits)
        _validate_dataset(name, data)
        print(f"{name}: {len(data['y'])} nodes, {data['x'].shape[1]} features, "
              f"{len(data['splits'])} splits ready", flush=True)
        del data
    preflight_inputs(names, root)


def _validate_dataset(name, data):
    x, y, edges = data["x"], data["y"], data["edge_index"]
    if x.ndim != 2 or len(x) != len(y) or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"{name}: inconsistent feature, label, or edge tensor shapes")
    # Chunk the feature check to keep the products validation allocation small.
    if any(not bool(torch.isfinite(block).all()) for block in x.split(100_000)):
        raise ValueError(f"{name}: features contain NaN or infinity")
    if len(y) == 0 or int(y.min()) < 0:
        raise ValueError(f"{name}: invalid labels")
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= len(y)):
        raise ValueError(f"{name}: invalid edge endpoints")
    for split in data["splits"]:
        combined = torch.cat(list(split.values()))
        if (combined.numel() == 0 or int(combined.min()) < 0 or int(combined.max()) >= len(y)
                or combined.unique().numel() != combined.numel()):
            raise ValueError(f"{name}: invalid or overlapping split indices")


#Download CS-TAG graphs and RoBERTa features.
def download_cstag(name, root):
    try:
        import dgl
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError("CS-TAG conversion needs dgl and huggingface_hub. Alternatively supply graph.pt") from error
    source = CSTAG[name]
    graph_path = hf_hub_download("Sherirto/CSTAG", f"{source}/{source}.pt", repo_type="dataset")
    feature_path = hf_hub_download("Sherirto/CSTAG", f"{source}/Feature/{source}_roberta_base_512_cls.npy",
                                   repo_type="dataset")
    graph = dgl.load_graphs(graph_path)[0][0]
    label = next(key for key in ("label", "labels", "y") if key in graph.ndata)
    x = torch.from_numpy(np.load(feature_path)).float()
    y = graph.ndata[label].long().reshape(-1)
    directory = Path(root) / name
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x, "y": y, "edge_index": torch.stack(graph.edges()).long()}, directory / "graph.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["all"], choices=("all", *DATASETS))
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--data-source", "--source", type=Path, help="Released paper-data.zip or directory of processed inputs")
    parser.add_argument("--cstag", action="store_true", help="Download and convert CS-TAG inputs from Hugging Face")
    args = parser.parse_args()
    names = DATASETS if "all" in args.datasets else args.datasets
    if args.cstag:
        for name in names:
            if name in CSTAG and not (args.root / name / "graph.pt").exists():
                download_cstag(name, args.root)
    prepare_datasets(names, args.root, source=args.data_source, splits=args.splits)


if __name__ == "__main__":
    main()
