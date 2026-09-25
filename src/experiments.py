"""Run experiments on saved predictions."""

import argparse
import copy
import csv
import json
from pathlib import Path
from statistics import median

import optuna
import torch
import yaml

from .methods.baselines import LABEL_AWARE, predict
from .diagnostics import diagnostic_rows, metrics
from .selection import tune
from .seeds import study_seed

MAIN_DATASETS = ('wikics', 'cora-tag', 'pubmed-tag', 'tape-arxiv23', 'ogbn-arxiv',
                 'ogbn-products', 'ele-photo', 'ele-computers', 'books-history')
CALIBRATION_DATASETS = MAIN_DATASETS[:6]
STAGES = ('main', 'external', 'mass', 'transfer', 'depth', 'energy',
          'calibration', 'per_node', 'timing')
CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'paper.yaml'


def unit_info(folder):
    return dict(dataset=folder.parents[2].name, backbone=folder.parents[1].name,
                split=int(folder.parent.name.split('_')[1]), seed=int(folder.name.split('_')[1]))


def context_info(path):
    sigma, draw = path.name.removesuffix('.pt')[1:].split('_d')
    return float(sigma), int(draw)


def context_files(folder, stage='main'):
    paths = sorted(folder.glob('s*_d*.pt'), key=context_info)
    if stage not in ('main', 'transfer'):
        paths = [p for p in paths if context_info(p)[0] in (0, 2)]
    if stage in ('external', 'mass') and unit_info(folder)['dataset'] == 'ogbn-products':
        paths = [p for p in paths if context_info(p)[0] == 0 or context_info(p)[1] in (0, 1)]
    if stage in ('per_node', 'timing'):
        paths = [p for p in paths if context_info(p)[0] == 0 or context_info(p)[1] == 0]
    if stage in ('energy', 'timing'):
        paths = [p for p in paths if context_info(p)[0] == 2]
    return paths


def load_context(path, graph, split, device='cpu'):
    saved = torch.load(path, weights_only=True, map_location=device)
    labels = graph['y'].to(device)
    indices = {name: index.to(device) for name, index in split.items()}
    return {'q': saved['Q'], 'logits': saved['Z'], 'edges': graph['edge_index'].to(device),
            'y': labels, 'splits': indices, 'train_labels': labels[indices['train_index']],
            'metadata': saved['metadata']}


def method_prediction(method, params, data, operators):
    return predict(method, data['q'], data['edges'], params, logits=data['logits'],
                   train_index=data['splits']['train_index'], train_labels=data['train_labels'],
                   operators=operators)


def score(probability, data):
    return metrics(probability, data['y'], data['splits']['test_index'])


def write_results(folder, stage, rows):
    groups = {}
    for row in rows:
        group = 'label_aware' if row['method'] in LABEL_AWARE else 'label_free'
        if row['method'] == 'backbone':
            group = 'backbone'
        item = dict(row, access=group)
        item = {k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                for k, v in item.items()}
        groups.setdefault(group, []).append(item)
    for group, records in groups.items():
        columns = list(dict.fromkeys(key for row in records for key in row))
        output = folder / f'{stage}.{group}.csv'
        temporary = output.with_suffix('.csv.tmp')
        with temporary.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
        temporary.replace(output)


def methods_for(stage, backbone, dataset):
    if stage == 'main':
        return ['appnp', 'ppr', 'pts', 'cs', 'cs_pts'] if backbone == 'mlp' else ['appnp', 'pts']
    if stage == 'external':
        return ['lame'] if dataset == 'ogbn-products' else ['lame', 'graph_tv']
    if stage == 'mass':
        return ['ppr_rn', 'pts_rn']
    return []


def select_methods(folder, stage, graph, split, config, device, operators):
    info = unit_info(folder)
    output = folder / f'{stage}.selections.json'
    selections = json.loads(output.read_text()) if output.exists() else {}
    search = copy.deepcopy(config['search'])
    if info['dataset'] == 'ogbn-products':
        search['max_steps'] = min(search['max_steps'], 50)
    for path in context_files(folder, stage):
        saved = selections.setdefault(path.stem, {})
        pending = [m for m in methods_for(stage, info['backbone'], info['dataset']) if m not in saved]
        if not pending:
            continue
        data = load_context(path, graph, split, device)
        val = data['splits']['val_index']

        def evaluate(method, params):
            return method_prediction(method, params, data, operators)

        for method in pending:
            sigma, draw = context_info(path)
            tag = config.get('optuna_seed_tag', 'optuna_board_v2')
            if info['backbone'] != 'mlp':
                tag += '_' + info['backbone']
            original = {'pts': 'potts', 'pts_rn': 'potts_norm',
                        'ppr_rn': 'ppr_norm', 'cs_pts': 'cs_potts',
                        'graph_tv': 'graphtv'}.get(method, method)
            search['seed'] = study_seed(info['dataset'], info['split'], info['seed'],
                                        sigma, draw, original, tag=tag)
            params, probability, _, history = tune(method, evaluate, config['methods'][method],
                                                    val, data['y'][val], search)
            accuracy = float(probability[val].argmax(1).eq(data['y'][val]).double().mean())
            saved[method] = {'parameters': params, 'validation_accuracy': accuracy, 'search': history}
            temporary = output.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(selections, indent=2) + '\n')
            temporary.replace(output)
            print(f"  {stage}: {path.stem} {method}, validation {accuracy:.4f}", flush=True)
    return selections


def selected_parameters(selections, context):
    return {name: record['parameters'] for name, record in selections[context].items()}


def refinement_rows(folder, stage, selections, main, graph, split, device, operators):
    rows, info = [], unit_info(folder)
    for path in context_files(folder, stage):
        data = load_context(path, graph, split, device)
        sigma, draw = context_info(path)
        params = selected_parameters(selections, path.stem)
        if stage == 'external':
            params.update({m: main[path.stem][m]['parameters'] for m in ('appnp', 'pts')})
        if stage == 'mass':
            params.update({m: main[path.stem][m]['parameters'] for m in ('ppr', 'pts')})
        params = {'anchor': {}, **params}
        for method, settings in params.items():
            probability, _ = method_prediction(method, settings, data, operators)
            rows.append({**info, 'sigma': sigma, 'draw': draw, 'method': method,
                         'parameters': settings, **score(probability, data)})
            if stage == 'main' and method == 'pts':
                off = dict(settings, eta=0.)
                probability, _ = method_prediction('pts', off, data, operators)
                rows.append({**info, 'sigma': sigma, 'draw': draw, 'method': 'reaction_off',
                             'parameters': off, **score(probability, data)})
    return rows


def transfer_rows(folder, selections, graph, split, device, operators):
    rows, info = [], unit_info(folder)
    paths = context_files(folder)
    for target in paths:
        data = load_context(target, graph, split, device)
        sigma, draw = context_info(target)
        for source in paths:
            sigma_selected, draw_selected = context_info(source)
            if draw_selected != draw and sigma_selected != 0 and sigma != 0:
                continue
            for method in ('appnp', 'ppr', 'pts'):
                if method not in selections[source.stem]:
                    continue
                params = selections[source.stem][method]['parameters']
                probability, _ = method_prediction(method, params, data, operators)
                rows.append({**info, 'sigma': sigma, 'draw': draw,
                             'sigma_selected': sigma_selected, 'draw_selected': draw_selected,
                             'method': method, 'parameters': params, **score(probability, data)})
    return rows


def applicable(stage, info):
    if stage in ('main', 'transfer'):
        return True
    if stage == 'timing':
        return info['split'] == 0 and info['seed'] == 0
    if info['backbone'] != 'mlp':
        return False
    if stage == 'depth':
        return info['dataset'] in MAIN_DATASETS
    if stage == 'energy':
        return info['dataset'] == 'wikics'
    if stage == 'calibration':
        return info['dataset'] in CALIBRATION_DATASETS
    if stage == 'per_node':
        return info['seed'] == 0 and info['dataset'] != 'ogbn-products'
    return True


#Get median settings for reproducibility
def median_settings(run, dataset):
    values = {}
    for path in (run / dataset / 'mlp').glob('split_*/seed_*/main.selections.json'):
        for name, context in json.loads(path.read_text()).items():
            if context_info(Path(name))[0] != 2:
                continue
            for method in ('appnp', 'ppr', 'pts', 'cs', 'cs_pts'):
                if method in context:
                    for key, value in context[method]['parameters'].items():
                        values.setdefault(method, {}).setdefault(key, []).append(value)
    return {method: {key: int(round(median(numbers))) if key.startswith('steps') else median(numbers)
                     for key, numbers in params.items()} for method, params in values.items()}


def forward_callback(folder, graph, data_root, device):
    from .backbones import load_checkpoint, preprocess
    from .data import load_dataset
    info = unit_info(folder)
    dataset = load_dataset(info['dataset'], data_root, splits=info['split'] + 1)
    split = torch.load(folder / 'split.pt', weights_only=True)
    saved = torch.load(folder / 'checkpoint.pt', weights_only=True)
    x = preprocess(dataset['x'], split['train_index'], saved['preprocessing']).to(device)
    model = load_checkpoint(folder / 'checkpoint.pt', graph['edge_index'], device)
    forward = lambda: model(x)
    forward.training_seconds = saved.get('training_seconds')
    return forward


def run_experiments(run, stages=('all',), *, config=None, device='cpu', threads=8,
                    data_root=Path('data'), datasets=None, backbones=None):
    run = Path(run)
    config = yaml.safe_load(CONFIG.read_text()) if config is None else config
    stages = list(STAGES) if 'all' in stages else list(stages)
    torch.set_num_threads(threads)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    units = [p for p in sorted(run.glob('*/*/split_*/seed_*')) if (p / 'checkpoint.pt').exists()]
    units = [p for p in units if (not datasets or unit_info(p)['dataset'] in datasets)
             and (not backbones or unit_info(p)['backbone'] in backbones)]
    if not units:
        raise FileNotFoundError('No trained backbones found. Run train.py first.')
    for folder in units:
        info = unit_info(folder)
        print('/'.join(str(info[k]) for k in ('dataset', 'backbone', 'split', 'seed')), flush=True)
        wanted = [stage for stage in stages if applicable(stage, info)]
        if not wanted:
            continue
        graph = torch.load(folder.parents[2] / 'graph.pt', weights_only=True)
        split = torch.load(folder / 'split.pt', weights_only=True)
        operators = {}
        needs_main = any(stage not in ('depth', 'energy') for stage in wanted)
        main = select_methods(folder, 'main', graph, split, config, device, operators) if needs_main else {}
        if needs_main:
            write_results(folder, 'main', refinement_rows(folder, 'main', main, main,
                                                         graph, split, device, operators))
        for stage in wanted:
            if stage in ('main', 'timing'):
                continue
            if stage in ('external', 'mass'):
                selections = select_methods(folder, stage, graph, split, config, device, operators)
                rows = refinement_rows(folder, stage, selections, main, graph, split, device, operators)
            elif stage == 'transfer':
                rows = transfer_rows(folder, main, graph, split, device, operators)
            else:
                rows = []
                for path in context_files(folder, stage):
                    data = load_context(path, graph, split, device)
                    parameters = {'anchor': {}, **selected_parameters(main, path.stem)} if main else {'anchor': {}}
                    sigma, draw = context_info(path)
                    rows.extend({**info, 'sigma': sigma, 'draw': draw, **row}
                                for row in diagnostic_rows(stage, data, {'methods': parameters}))
            if not rows:
                print(f'  {stage}: no applicable prediction contexts', flush=True)
            write_results(folder, stage, rows)
        del operators, graph
    if 'timing' in stages:
        for folder in units:
            info = unit_info(folder)
            if not applicable('timing', info):
                continue
            graph = torch.load(folder.parents[2] / 'graph.pt', weights_only=True)
            split = torch.load(folder / 'split.pt', weights_only=True)
            paths = context_files(folder, 'timing')
            if not paths:
                raise FileNotFoundError('Timing needs sigma=2, draw=0 predictions.')
            timing_device = device if info['dataset'].startswith('ogbn-') else 'cpu'
            torch.set_num_threads(8)
            data = load_context(paths[0], graph, split, timing_device)
            params = median_settings(run, info['dataset']) if info['backbone'] == 'mlp' else {}
            forward = forward_callback(folder, graph, data_root, timing_device)
            records = diagnostic_rows('timing', data, {'methods': params}, forward=forward, timing_params=params)
            write_results(folder, 'timing', [{**info, 'sigma': 2., 'draw': 0, **row} for row in records])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--experiment', nargs='+', choices=['all', *STAGES], default=['all'])
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--backbones', nargs='+', choices=['mlp', 'gcn', 'sage'])
    parser.add_argument('--trials', type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.trials is not None:
        config['search']['trials'] = args.trials
    run_experiments(args.run, args.experiment, config=config, device=args.device, threads=args.threads,
                    data_root=args.data_root, datasets=args.datasets, backbones=args.backbones)


if __name__ == '__main__':
    main()
