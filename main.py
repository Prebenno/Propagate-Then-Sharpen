"""Run a pretrained example, selected experiments, or the complete paper pipeline."""

import argparse
from copy import copy
from pathlib import Path

import yaml

from src.experiments import CONFIG, STAGES, run_experiments
from train import add_arguments, generate


PAPER_STEPS = ('data', 'train', 'predictions', 'main', 'external', 'mass', 'transfer',
               'depth', 'energy', 'calibration', 'per_node', 'timing')


def paper_run(args, config):
    """Run every stage of the recorded protocol on the full dataset cohort."""
    args.datasets = config['datasets']['main'] + config['datasets']['controls']
    args.backbones = config['architectures']
    args.seeds = config['units']['model_seeds']
    args.splits = config['units']['splits']
    args.draws = list(range(config['units']['draws']))
    args.sigmas = config['corruption']['sigmas']
    args.device = args.device or 'cuda'
    args.threads = args.threads or 8
    output = args.output
    from src.data import prepare_datasets
    prepare_datasets(args.datasets, args.data_root, source=args.data_source, splits=args.splits)
    output.mkdir(parents=True, exist_ok=True)
    for stage in PAPER_STEPS[1:]:
        print(f'\n[{stage}]', flush=True)
        if stage in ('train', 'predictions'):
            generation = copy(args)
            generation.sigmas = [0.] if stage == 'train' else args.sigmas
            generate(generation)
        elif stage == 'timing':
            for names, device in (([d for d in args.datasets if not d.startswith('ogbn-')], 'cpu'),
                                  ([d for d in args.datasets if d.startswith('ogbn-')],
                                   args.device if args.device.startswith('cuda') else 'cuda')):
                run_experiments(output, [stage], config=config, device=device, threads=8,
                                data_root=args.data_root, datasets=names, backbones=args.backbones)
        else:
            run_experiments(output, [stage], config=config, device=args.device,
                            threads=args.threads, data_root=args.data_root,
                            datasets=args.datasets, backbones=args.backbones)
    print(f'Full pipeline complete: {output}', flush=True)


def parser():
    result = add_arguments(argparse.ArgumentParser(description=__doc__))
    result.set_defaults(device=None, threads=None)
    result.add_argument('--config', type=Path, default=CONFIG)
    result.add_argument('--trials', type=int, help='Override the validation trial budget')
    result.add_argument('--experiment', nargs='+', choices=['all', *STAGES])
    result.add_argument('--paper', action='store_true', help='Run the complete recorded paper protocol')
    result.add_argument('--data-source', type=Path, help='Released paper-data.zip or extracted data directory')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if args.trials is not None:
        config['search']['trials'] = args.trials
    if args.paper:
        paper_run(args, config)
        return
    args.device = args.device or 'cpu'
    args.threads = args.threads or 1
    args.experiment = args.experiment or (['main'] if args.pretrained is not None else ['all'])
    if args.data_source is not None:
        from src.data import DATASETS, prepare_datasets
        prepare_datasets(DATASETS if 'all' in args.datasets else args.datasets,
                         args.data_root, source=args.data_source, splits=args.splits)
    generate(args)
    run_experiments(args.output, args.experiment, config=config, device=args.device,
                    threads=args.threads, data_root=args.data_root,
                    datasets=None if 'all' in args.datasets else args.datasets,
                    backbones=args.backbones)


if __name__ == '__main__':
    main()
