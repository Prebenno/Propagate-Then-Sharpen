"""List the pretrained models and saved Q/Z files."""

import argparse
from pathlib import Path

from src.pretrained import DEFAULT_ROOT, available_checkpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--datasets", nargs="+")
    args = parser.parse_args()
    paths = available_checkpoints(args.root)
    if args.datasets:
        paths = [path for path in paths if path.relative_to(args.root).parts[0] in args.datasets]
    print("Dataset          Backbone  Split  Seed  Q/Z files")
    for path in paths:
        dataset, backbone, split, seed, _ = path.relative_to(args.root).parts
        contexts = len(list(path.parent.glob("s*_d*.pt")))
        print(f"{dataset:<16} {backbone:<9} {split[6:]:>5} {seed[5:]:>5} {contexts:>10}")
    print(f"{len(paths)} pretrained models.")


if __name__ == "__main__":
    main()
