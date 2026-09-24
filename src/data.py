"""Load graphs and prepare splits."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch_geometric.datasets import HeterophilousGraphDataset, WikiCS



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


def _random_splits(n, count):
    train, val = round(.6 * n), round(.2 * n)
    result = []
    for seed in range(1, count + 1):
        order = np.random.default_rng(seed).permutation(n)
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
        return data, _random_splits(len(data["y"]), count)
    ogb_directory = directory / name.replace("-", "_")
    cached = ogb_directory / "processed/geometric_data_processed.pt"
    if cached.exists():
        data = _read(cached)
        split_name = "time" if name == "ogbn-arxiv" else "sales_ranking"
        indices = {role: torch.from_numpy(np.loadtxt(ogb_directory / "split" / split_name / f"{role}.csv.gz",
                                                    delimiter=",", dtype=np.int64)).reshape(-1)
                   for role in ("train", "valid", "test")}
    else:
        from ogb.nodeproppred import PygNodePropPredDataset
        from torch_geometric.data import Data
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
        from torch_geometric.data.storage import GlobalStorage
        with torch.serialization.safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]):
            dataset = PygNodePropPredDataset(name=name, root=str(directory))
        data, indices = _extract(dataset[0]), dataset.get_idx_split()
    return data, [{f"{role}_index": indices["valid" if role == "val" else role].long()
                   for role in ("train", "val", "test")}]


def load_dataset(name, root, splits=10):
    root = Path(root)
    if name in CUSTOM:
        path = root / name / "graph.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Supply {FEATURES[name]} in a graph.pt with x, y and edge_index.")
        data = _read(path)
        indices = _random_splits(len(data["y"]), splits)
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
    parser.add_argument("--cstag", action="store_true", help="Download and convert CS-TAG inputs from Hugging Face")
    args = parser.parse_args()
    names = DATASETS if "all" in args.datasets else args.datasets
    missing = []
    for name in names:
        path = args.root / name / "graph.pt"
        if name in CUSTOM and not path.exists():
            if args.cstag and name in CSTAG:
                download_cstag(name, args.root)
            else:
                missing.append(f"{path}: {FEATURES[name]}")
                continue
        data = load_dataset(name, args.root, args.splits)
        print(f"{name}: {len(data['y'])} nodes, {data['x'].shape[1]} features, {len(data['splits'])} splits")
    if missing:
        parser.exit(1, "Missing custom inputs:\n" + "\n".join(missing) + "\n")


if __name__ == "__main__":
    main()
