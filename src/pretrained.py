"""Paths for the pretrained models and their saved predictions."""

from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "pretrained"


def unit_directory(root, dataset, backbone, split, seed):
    return Path(root) / dataset / backbone / f"split_{split}" / f"seed_{seed}"


def available_checkpoints(root=DEFAULT_ROOT):
    return sorted(Path(root).glob("*/*/split_*/seed_*/checkpoint.pt"))
