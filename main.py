"""Train backbones, run experiments and save tables."""

import argparse
from pathlib import Path

import yaml

from src.experiments import CONFIG, STAGES, run_experiments
from train import add_arguments, generate
from report import build


def main():
    parser = add_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--trials', type=int, help='Override the 250-trial validation searches')
    parser.add_argument('--experiment', nargs='+', choices=['all', *STAGES], default=['all'])
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.trials is not None:
        config['search']['trials'] = args.trials
    generate(args)
    run_experiments(args.output, args.experiment, config=config, device=args.device,
                    threads=args.threads, data_root=args.data_root,
                    datasets=None if 'all' in args.datasets else args.datasets,
                    backbones=args.backbones)
    build(args.output)


if __name__ == '__main__':
    main()
