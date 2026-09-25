"""Run a pretrained example, selected experiments, or the complete paper pipeline."""

import argparse
from contextlib import contextmanager
from copy import copy
from pathlib import Path

import torch
import yaml

from src.experiments import CONFIG, STAGES, check_settings, run_experiments, validate_recorded_settings
from train import add_arguments, generate
from report import build


PAPER_STEPS = ('data', 'train', 'predictions', 'main', 'external', 'mass', 'transfer',
               'depth', 'energy', 'calibration', 'per_node', 'timing', 'report')


def write_yaml(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(yaml.safe_dump(value, sort_keys=False))
    temporary.replace(path)


@contextmanager
def run_lock(output):
    """Prevent two full runs from updating the same output directory."""
    import fcntl
    with (output / '.run.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another process is using this output directory') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def paper_arguments(args, config):
    """Resolve the full protocol; subsets use the ordinary main.py interface."""
    validate_recorded_settings(config)
    if config['search']['trials'] != 250:
        raise ValueError('--paper requires the recorded 250-trial budget; use individual runs for checks')
    if args.pretrained is not None or args.epochs is not None:
        raise ValueError('--paper trains from scratch; --pretrained and --epochs are for individual runs')
    datasets = config['datasets']['main'] + config['datasets']['controls']
    expected = {'datasets': datasets, 'backbones': config['architectures'],
                'seeds': config['units']['model_seeds'], 'splits': config['units']['splits'],
                'draws': list(range(config['units']['draws'])),
                'sigmas': config['corruption']['sigmas']}
    for key, value in expected.items():
        supplied = getattr(args, key)
        default = {'datasets': ['all'], 'backbones': ['mlp', 'gcn', 'sage'],
                   'splits': 10, 'draws': [0, 1, 2]}.get(key)
        if supplied is not None and supplied != default and supplied != value:
            raise ValueError(f'--paper requires {key}={value}; use main.py without --paper for subsets')
        setattr(args, key, value)
    if args.experiment not in (None, ['all']):
        raise ValueError('--paper runs every stage; use main.py without --paper for individual stages')
    args.device = args.device or 'cuda'
    args.threads = args.threads or 8
    args.experiment = ['all']
    return args


def print_plan(args, config):
    units = sum(1 if d.startswith('ogbn-') else args.splits for d in args.datasets)
    units *= len(args.backbones) * len(args.seeds)
    print(f'Full paper: {len(args.datasets)} datasets, {units} backbone units; '
          f'{config["search"]["trials"]} validation trials per search')
    print(f'Model seeds: {args.seeds}; noisy draws: {args.draws}; clean draw: -1')
    print(f'Severities: {args.sigmas}; output: {args.output}')
    descriptions = {
        'data': 'prepare exact released inputs and official ogbn-products',
        'train': 'train each backbone once and save clean Q/Z',
        'predictions': 'generate corrupted Q/Z from the same frozen checkpoints',
        'main': 'main comparisons and validation selection',
        'external': 'LAME-Graph and Graph-TV endpoint comparisons',
        'mass': 'mass-reset endpoint comparisons',
        'transfer': 'reuse selected settings across severities',
        'depth': 'clean and severe depth sweeps',
        'energy': 'WikiCS energy and validation-accuracy traces',
        'calibration': 'six-dataset calibration study',
        'per_node': 'confidence and degree strata, excluding products',
        'timing': 'CPU/8 threads for non-OGB; CUDA for OGB; fixed-100 and selected settings',
        'report': 'complete CSV tables and plots, including the analytical example',
    }
    for index, stage in enumerate(PAPER_STEPS, 1):
        print(f'{index:2}. {stage}: {descriptions[stage]}')
    print('Products: two external/mass draws; no Graph-TV. Controls excluded from main means.')


def paper_run(args, config):
    args = paper_arguments(args, config)
    print_plan(args, config)
    if args.dry_run:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('The full paper includes CUDA timing for OGB. Run on a CUDA machine; --dry-run needs no GPU.')
    output = args.output
    settings = {'config': config, 'data_root': str(args.data_root.resolve()),
                'device': args.device, 'threads': args.threads}
    existing = output / 'run.yaml'
    occupied = output.exists() and any(output.iterdir())
    if occupied:
        if not args.resume:
            raise ValueError('Output directory is not empty. Use --resume for this run or choose a new --output.')
        if not existing.is_file() or yaml.safe_load(existing.read_text()) != settings:
            raise ValueError('Existing run settings differ or are missing; choose a new output directory.')
    check_settings(output, config)
    from src.data import prepare_datasets
    # Every dataset is prepared/validated before any backbone can be trained.
    prepare_datasets(args.datasets, args.data_root, source=args.data_source, splits=args.splits)
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        write_yaml(existing, settings)
        progress_path = output / 'progress.yaml'
        completed = set(yaml.safe_load(progress_path.read_text()).get('completed', [])) if progress_path.exists() else set()
        completed.add('data')
        for stage in PAPER_STEPS[1:]:
            # Always validate/regenerate reports when resuming, even after a complete run.
            if stage in completed and stage != 'report':
                print(f'{stage}: already complete', flush=True)
                continue
            print(f'\n[{stage}]', flush=True)
            if stage in ('train', 'predictions'):
                generation = copy(args)
                generation.sigmas = [0.] if stage == 'train' else args.sigmas
                generate(generation)
            elif stage == 'report':
                build(output, config=config, require_complete=True, plots=True)
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
            completed.add(stage)
            write_yaml(progress_path, {'completed': [s for s in PAPER_STEPS if s in completed]})
        print(f'Full pipeline complete: {output / "report"}', flush=True)


def parser():
    result = add_arguments(argparse.ArgumentParser(description=__doc__))
    result.set_defaults(device=None, threads=None)
    result.add_argument('--config', type=Path, default=CONFIG)
    result.add_argument('--trials', type=int, help='Override validation trials for an individual run')
    result.add_argument('--experiment', nargs='+', choices=['all', *STAGES])
    result.add_argument('--paper', action='store_true', help='Run the complete recorded paper protocol')
    result.add_argument('--data-source', type=Path, help='Released paper-data.zip or extracted data directory')
    result.add_argument('--dry-run', action='store_true', help='Print the full pipeline without downloads or computation')
    result.add_argument('--resume', action='store_true', help='Continue a compatible full-paper run')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if args.trials is not None:
        if args.paper and args.trials != config['search']['trials']:
            raise ValueError('--paper uses the recorded trial budget; omit --paper for a quick check')
        config['search']['trials'] = args.trials
    if args.paper:
        paper_run(args, config)
        return
    if args.dry_run or args.resume:
        raise ValueError('--dry-run and --resume belong to run_all.sh / --paper')
    args.device = args.device or 'cpu'
    args.threads = args.threads or 1
    args.experiment = args.experiment or (['main'] if args.pretrained is not None else ['all'])
    check_settings(args.output, config)
    if args.data_source is not None:
        from src.data import DATASETS, prepare_datasets
        prepare_datasets(DATASETS if 'all' in args.datasets else args.datasets,
                         args.data_root, source=args.data_source, splits=args.splits)
    generate(args)
    run_experiments(args.output, args.experiment, config=config, device=args.device,
                    threads=args.threads, data_root=args.data_root,
                    datasets=None if 'all' in args.datasets else args.datasets,
                    backbones=args.backbones)
    build(args.output, config=config)


if __name__ == '__main__':
    main()
