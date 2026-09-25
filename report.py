"""Build CSV tables and figures from a new run (separate from archived paper tables)."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = ('wikics', 'cora-tag', 'pubmed-tag', 'tape-arxiv23', 'ogbn-arxiv',
        'ogbn-products', 'ele-photo', 'ele-computers', 'books-history')
UNIT = ['dataset', 'backbone', 'split', 'seed']
CONTEXT = UNIT + ['sigma', 'draw']
AXES = {
    'main': ['sigma', 'method'], 'external': ['sigma', 'method'],
    'mass': ['sigma', 'method'], 'transfer': ['sigma', 'sigma_selected', 'method'],
    'depth': ['sigma', 'method', 'alpha', 'K', 'eta'],
    'energy': ['sigma', 'method', 'K', 'phase', 'alpha', 'eta'],
    'calibration': ['sigma', 'method'],
    'per_node': ['sigma', 'method', 'variable', 'bucket', 'bucket_index'],
    'timing': ['method', 'mode', 'K', 'device', 'hardware', 'threads', 'dtype'],
}
METRICS = {
    'main': ['accuracy', 'nll', 'brier', 'ece'],
    'external': ['accuracy', 'nll', 'brier', 'ece'],
    'mass': ['accuracy', 'nll', 'brier', 'ece'],
    'transfer': ['accuracy', 'nll', 'brier', 'ece'],
    'depth': ['accuracy', 'val_accuracy'],
    'energy': ['dirichlet', 'gini', 'potts', 'kernel_mass', 'val_accuracy'],
    'calibration': ['temperature', 'raw_accuracy', 'cal_accuracy', 'raw_nll', 'cal_nll',
                    'raw_brier', 'cal_brier', 'raw_ece', 'cal_ece'],
    'per_node': ['accuracy'], 'timing': ['ms', 'ms_per_step', 'training_seconds'],
}

def read_results(run):
    collected = {}
    for path in sorted(Path(run).glob('*/*/split_*/seed_*/*.csv')):
        parts = path.name.split('.')
        if len(parts) != 3:
            continue
        stage, access, suffix = parts
        if stage not in AXES or suffix != 'csv':
            continue
        frame = pd.read_csv(path)
        if not frame.empty:
            validate_rows(frame, stage, access, path)
            collected.setdefault((stage, access), []).append(frame)
    return {key: pd.concat(frames, ignore_index=True) for key, frames in collected.items()}


#Average draws and seeds, then summarize splits.
def reduce(frame, axes, metrics):
    metrics = [name for name in metrics if name in frame]
    if frame.empty or not metrics:
        return pd.DataFrame()
    keys = ['dataset', 'backbone', 'access', *axes]
    values = frame.melt(id_vars=[*keys, 'split', 'seed'], value_vars=metrics,
                        var_name='metric', value_name='value').dropna(subset=['value'])
    rows = []
    for key, part in values.groupby([*keys, 'metric'], sort=False, dropna=False):
        seed_means = part.groupby(['split', 'seed']).value.mean()
        split_means = seed_means.groupby('split').mean()
        if str(key[0]).startswith('ogbn-') and len(split_means) == 1:
            sd, basis = part.value.std(ddof=1), 'seed_draw_sd'
        elif len(split_means) > 1:
            sd, basis = split_means.std(ddof=1), 'split_sd'
        else:
            sd, basis = np.nan, 'single_split_sd_unavailable'
        rows.append({**dict(zip([*keys, 'metric'], key)), 'mean': split_means.mean(),
                     'sd': sd, 'sd_basis': basis, 'n_splits': len(split_means),
                     'n_seeds': part.seed.nunique(), 'n_runs': len(part), 'n_datasets': 1})
    return pd.DataFrame(rows)


#Give each dataset equal weight
def macro(summary, roster=MAIN):
    if summary.empty:
        return summary.copy()
    part = summary[summary.dataset.isin(roster)]
    if part.empty:
        return part.copy()
    excluded = {'dataset', 'mean', 'sd', 'sd_basis', 'n_splits', 'n_seeds', 'n_runs', 'n_datasets'}
    keys = [name for name in summary if name not in excluded]
    rows = []
    for key, group in part.groupby(keys, sort=False, dropna=False):
        rows.append({**dict(zip(keys, key if isinstance(key, tuple) else (key,))),
                     'dataset': 'main_mean' if group.dataset.nunique() == len(roster) else 'partial_mean',
                     'mean': group['mean'].mean(), 'sd': np.nan,
                     'sd_basis': 'not_reported_for_macro', 'n_splits': np.nan,
                     'n_seeds': np.nan, 'n_runs': int(group.n_runs.sum()),
                     'n_datasets': group.dataset.nunique(), 'expected_datasets': len(roster)})
    return pd.DataFrame(rows)


#Compare methods on matching runs.
def paired(frame, left, right, extra=()):
    keys = CONTEXT + ['access', *extra]
    keys += [name for name in ('sigma_selected', 'draw_selected') if name in frame and name not in keys]
    a = frame[frame.method.eq(left)][keys + ['accuracy']]
    b = frame[frame.method.eq(right)][keys + ['accuracy']]
    together = a.merge(b, on=keys, suffixes=('_left', '_right'), validate='one_to_one')
    together['accuracy'] = together.accuracy_left - together.accuracy_right
    together['method'] = f'{left} - {right}'
    return together.drop(columns=['accuracy_left', 'accuracy_right'])


def contrasts(frame, pairs, axes, extra=()):
    records = [paired(frame, left, right, extra) for left, right in pairs]
    records = [part for part in records if not part.empty]
    if not records:
        return pd.DataFrame()
    result = reduce(pd.concat(records, ignore_index=True), axes, ['accuracy'])
    result['metric'] = 'accuracy_difference'
    return result


def common_contexts(frame, methods):
    part = frame[frame.method.isin(methods)]
    keys = CONTEXT + ['access']
    complete = part.groupby(keys).method.nunique().eq(len(methods))
    return part.merge(complete[complete].reset_index()[keys], on=keys)


def selected_parameters(run):
    """Independently computed medians and reaction-off frequency at every severity."""
    rows = []
    for path in sorted(Path(run).glob('*/*/split_*/seed_*/*.selections.json')):
        folder = path.parent
        stage = path.name.split('.')[0]
        dataset, backbone = folder.parents[2].name, folder.parents[1].name
        for context, methods in json.loads(path.read_text()).items():
            sigma = float(context.split('_d')[0][1:])
            for method, record in methods.items():
                access = 'label_aware' if method in ('cs', 'cs_pts') else 'label_free'
                params = record['parameters']
                for parameter, value in params.items():
                    if isinstance(value, (float, int)):
                        rows.append({'dataset': dataset, 'backbone': backbone, 'stage': stage,
                                     'sigma': sigma, 'method': method, 'access': access,
                                     'parameter': parameter, 'value': value,
                                     'off': float(params['eta'] == 0) if 'eta' in params else np.nan})
    if not rows:
        return pd.DataFrame()
    keys = ['dataset', 'backbone', 'stage', 'sigma', 'method', 'access', 'parameter']
    result = pd.DataFrame(rows).groupby(keys, as_index=False).agg(
        median=('value', 'median'), n_contexts=('value', 'size'), off_fraction=('off', 'mean'))
    result['off_pct'] = 100 * result.off_fraction
    return result


def save_table(frame, directory, name):
    if not frame.empty:
        frame.to_csv(directory / f'{name}.csv', index=False)


def build(run, output=None, config=None, require_complete=False, plots=True):
    run = Path(run)
    output = Path(output) if output is not None else run / 'report'
    if config is None and (run / 'refinement.yaml').exists():
        import yaml
        config = yaml.safe_load((run / 'refinement.yaml').read_text())
    frames = read_results(run)
    if require_complete:
        validate_complete(run, frames, config)
    tables = output / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    for (stage, access), frame in frames.items():
        frame = frame.copy()
        if stage == 'calibration':
            frame['accuracy_drift_pp'] = 100 * (frame.cal_accuracy - frame.raw_accuracy)
        summary = reduce(frame, AXES[stage], METRICS[stage] + (['accuracy_drift_pp'] if stage == 'calibration' else []))
        save_table(summary, tables, f'{stage}.{access}')
        roster = (MAIN[:6] if stage == 'calibration' else ('wikics',) if stage == 'energy'
                  else tuple(name for name in MAIN if name != 'ogbn-products') if stage == 'per_node'
                  else MAIN)
        save_table(macro(summary, roster), tables, f'{stage}.{access}.main_mean')
        if stage == 'timing' and access == 'label_free':
            save_table(runtime_table(frame), tables, 'runtime_fixed_100.label_free')
        if stage == 'main':
            if access == 'label_free':
                endpoint = summary[summary.sigma.isin([0, 2]) & summary.metric.eq('accuracy')
                                   & summary.method.isin(['anchor', 'appnp', 'ppr', 'pts'])]
                save_table(pd.concat([endpoint, macro(endpoint)], ignore_index=True), tables, 'table1.label_free')
                comparisons = contrasts(frame, [('ppr', 'appnp'), ('pts', 'ppr'),
                                                 ('pts', 'reaction_off'), ('pts', 'appnp')], AXES[stage])
                save_table(pd.concat([comparisons, macro(comparisons)], ignore_index=True), tables, 'table2.label_free')
            else:
                comparison = contrasts(frame, [('cs_pts', 'cs')], AXES[stage])
                save_table(pd.concat([comparison, macro(comparison)], ignore_index=True), tables, 'cs_difference.label_aware')
        elif stage == 'external':
            for cohort, methods, roster in (
                    ('graph_tv_8', ['anchor', 'appnp', 'pts', 'lame', 'graph_tv'], MAIN[:5] + MAIN[6:]),
                    ('lame_9', ['anchor', 'appnp', 'pts', 'lame'], MAIN)):
                matched = common_contexts(frame[frame.dataset.isin(roster)], methods)
                result = reduce(matched, AXES[stage], ['accuracy'])
                averaged = macro(result, roster)
                save_table(pd.concat([result, averaged], ignore_index=True), tables, f'external.{cohort}.{access}')
                pairs = [('pts', method) for method in methods if method != 'pts']
                difference = contrasts(matched, pairs, AXES[stage])
                save_table(pd.concat([difference, macro(difference, roster)], ignore_index=True), tables, f'external.{cohort}.differences.{access}')
        elif stage == 'mass':
            comparison = contrasts(frame, [('pts', 'pts_rn'), ('ppr', 'ppr_rn')], AXES[stage])
            save_table(pd.concat([comparison, macro(comparison)], ignore_index=True), tables, f'mass.differences.{access}')
        elif stage == 'transfer':
            comparison = contrasts(frame, [('pts', 'appnp'), ('pts', 'ppr')], AXES[stage])
            save_table(pd.concat([comparison, macro(comparison)], ignore_index=True), tables, f'transfer.differences.{access}')
        elif stage == 'per_node':
            comparison = contrasts(frame, [('pts', 'appnp')], AXES[stage],
                                   extra=['variable', 'bucket', 'bucket_index'])
            roster = tuple(dataset for dataset in MAIN if dataset != 'ogbn-products')
            save_table(pd.concat([comparison, macro(comparison, roster)], ignore_index=True), tables, 'per_node.differences.label_free')
    selected = selected_parameters(run)
    if not selected.empty:
        for access, part in selected.groupby('access', sort=False):
            save_table(part, tables, f'selected_parameters.{access}')
    dataset_table, size_table = model_and_dataset_tables(run)
    save_table(dataset_table, tables, 'datasets')
    save_table(size_table, tables, 'model_sizes')
    if config is not None:
        save_table(search_table(config), tables, 'search_spaces')
    scope = 'complete declared paper cohort' if require_complete else 'partial / demonstration (full coverage not checked)'
    (output / 'README.md').write_text(
        f'# Generated results\n\nScope: **{scope}**.\n\n'
        'Accuracy and differences in the CSVs are fractions; figure axes use percent or percentage points. '
        'Temperature-calibration accuracy_drift_pp is already in percentage points. '
        'Means average draws within seeds, seeds within splits, then datasets equally. '
        'SD uses splits; single-split OGB results use descriptive seed/draw SD. '
        'A partial_mean is never a complete main-dataset mean. Label-aware tables are separate.\n\n'
        'Archived paper tables under results/paper are separate from these newly generated outputs. '
        'Figures are regenerated from these CSVs; no manuscript compilation is required.\n')
    if plots:
        from src.plotting import build as plot
        plot(output, partial=not require_complete)
    print(f'Tables: {tables}')
    return output


def validate_rows(frame, stage, access, path):
    """Reject mixed access budgets, duplicate observations and malformed values."""
    required = CONTEXT + AXES[stage] + ['access']
    if missing := sorted(set(required) - set(frame)):
        raise ValueError(f'{path}: missing result columns {missing}')
    expected_access = frame.method.map(lambda method: 'label_aware' if method in ('cs', 'cs_pts')
                                       else 'backbone' if method == 'backbone' else 'label_free')
    if not frame.access.eq(access).all() or not expected_access.eq(access).all():
        raise ValueError(f'{path}: mixed or incorrect method access budgets')
    identity = list(dict.fromkeys(required + (['draw_selected'] if stage == 'transfer' else [])))
    if frame[identity].isna().any().any() or frame.duplicated(identity).any():
        raise ValueError(f'{path}: null or duplicate scientific observation identity')
    if stage == 'timing':
        fixed = frame[frame['mode'].eq('fixed_100')]
        if not fixed.empty and (not fixed.K.eq(100).all() or 'ms_per_step' not in fixed or
                                not np.allclose(fixed.ms_per_step, fixed.ms / 100)):
            raise ValueError(f'{path}: fixed-100 timing must use K=100 and ms_per_step=ms/100')
    for key in ('accuracy', 'val_accuracy', 'raw_accuracy', 'cal_accuracy'):
        if key in frame and not frame[key].between(0, 1).all():
            raise ValueError(f'{path}: invalid {key}; expected a finite fraction in [0, 1]')


def validate_complete(run, frames, config=None):
    """Require every declared unit, prediction and stage/method/context before reporting."""
    import yaml
    from src.experiments import STAGES, applicable
    from src.diagnostics import ALPHAS, DEPTHS
    if config is None:
        config = yaml.safe_load((Path(__file__).parent / 'configs/paper.yaml').read_text())
    ds = config.get('datasets', {'main': list(MAIN), 'controls': ['roman-empire', 'amazon-ratings']})
    datasets = ds['main'] + ds['controls']
    architectures = config.get('architectures', ['mlp', 'gcn', 'sage'])
    units = config.get('units', {})
    seeds = units.get('model_seeds', [0, 1, 2])
    sigmas = config.get('corruption', {}).get('sigmas', [0, .5, 1, 1.5, 2])
    draws = list(range(units.get('draws', 3)))
    contexts = [(0., -1)] + [(float(s), d) for s in sigmas if s > 0 for d in draws]
    grouped = {}
    for (stage, _), frame in frames.items():
        for key, part in frame.groupby(UNIT, sort=False):
            grouped.setdefault((stage, *key), []).append(part)
    problems, expected_units = [], set()
    for dataset in datasets:
        if not (run / dataset / 'graph.pt').is_file():
            problems.append(f'{dataset}: graph.pt missing')
        for backbone in architectures:
            for split in range(1 if dataset.startswith('ogbn-') else units.get('splits', 10)):
                for seed in seeds:
                    key = (dataset, backbone, split, seed)
                    expected_units.add(key)
                    info = dict(zip(UNIT, key))
                    folder = run / dataset / backbone / f'split_{split}' / f'seed_{seed}'
                    files = ['checkpoint.pt', 'split.pt', *[f's{s:g}_d{d}.pt' for s, d in contexts]]
                    if missing := [name for name in files if not (folder / name).is_file()]:
                        problems.append(f'{dataset}/{backbone}/{split}/{seed}: {len(missing)} missing frozen input files')
                    for stage in STAGES:
                        if not applicable(stage, info):
                            continue
                        parts = grouped.get((stage, *key), [])
                        if not parts:
                            problems.append(f'{dataset}/{backbone}/{split}/{seed}: missing {stage}')
                            continue
                        frame = pd.concat(parts, ignore_index=True)
                        pairs = contexts if stage in ('main', 'transfer') else [(s, d) for s, d in contexts if s in (0, 2)]
                        if stage in ('external', 'mass') and dataset == 'ogbn-products':
                            pairs = [(s, d) for s, d in pairs if d in (-1, 0, 1)]
                        if stage in ('per_node', 'timing'):
                            pairs = [(s, d) for s, d in pairs if d in (-1, 0)]
                        if stage in ('energy', 'timing'):
                            pairs = [(s, d) for s, d in pairs if s == 2]
                        methods = {
                            'main': ['anchor', 'appnp', 'pts', 'reaction_off'] + (['ppr', 'cs', 'cs_pts'] if backbone == 'mlp' else []),
                            'external': ['anchor', 'appnp', 'pts', 'lame'] + ([] if dataset == 'ogbn-products' else ['graph_tv']),
                            'mass': ['anchor', 'ppr', 'pts', 'ppr_rn', 'pts_rn'],
                            'calibration': ['anchor', 'appnp', 'pts'],
                        }.get(stage)
                        columns = ['sigma', 'draw', 'method']
                        if methods is not None:
                            expected = {(s, d, m) for s, d in pairs for m in methods}
                        elif stage == 'transfer':
                            columns += ['sigma_selected', 'draw_selected']
                            methods = ['appnp', 'pts'] + (['ppr'] if backbone == 'mlp' else [])
                            expected = {(s, d, m, ss, dd) for s, d in pairs for ss, dd in pairs
                                        if dd == d or s == 0 or ss == 0 for m in methods}
                        elif stage == 'depth':
                            columns += ['alpha', 'K', 'eta']
                            variants = [('anchor', 0), ('appnp', 0), ('ppr', 0), ('logit_sharp', 16), ('pts', 16), ('pts', 200)]
                            expected = {(s, d, m, a, k, e) for s, d in pairs for m, e in variants for a in ALPHAS for k in DEPTHS}
                        elif stage == 'energy':
                            columns += ['K', 'phase']
                            phases = [(0, 'initial')] + [(k, phase) for k in range(1, 21) for phase in ('transport', 'reaction')]
                            expected = {(s, d, m, k, phase) for s, d in pairs for m in ('ppr', 'pts') for k, phase in phases}
                        elif stage == 'per_node':
                            columns += ['variable']
                            expected = {(s, d, m, v) for s, d in pairs for m in ('appnp', 'pts') for v in
                                        ('local_homophily', 'degree_quintile', 'anchor_confidence_quintile')}
                        else:
                            columns += ['mode']
                            variants = [('backbone', 'forward')]
                            if backbone == 'mlp':
                                variants += [(m, 'selected') for m in ('appnp', 'ppr', 'pts', 'cs', 'cs_pts')]
                                variants += [(m, 'fixed_100') for m in ('appnp', 'ppr', 'pts')]
                            expected = {(s, d, m, mode) for s, d in pairs for m, mode in variants}
                        actual = set(frame[columns].itertuples(index=False, name=None))
                        if actual != expected:
                            problems.append(f'{dataset}/{backbone}/{split}/{seed}: {stage} missing {len(expected - actual)}, unexpected {len(actual - expected)} contexts')
                        mandatory = [m for m in METRICS[stage] if m not in ('training_seconds', 'ms_per_step')]
                        if any(m not in frame or not np.isfinite(frame[m]).all() for m in mandatory):
                            problems.append(f'{dataset}/{backbone}/{split}/{seed}: {stage} lacks finite metrics')
    actual_units = {key[1:] for key in grouped}
    if actual_units - expected_units:
        problems.append(f'{len(actual_units - expected_units)} result units are outside the configured cohort')
    if problems:
        raise ValueError('Incomplete paper run; no full-paper report written.\n' + '\n'.join(problems[:20]) +
                         (f'\n... {len(problems)} issues total' if len(problems) > 20 else ''))


def model_and_dataset_tables(run):
    """Read one saved model per architecture and graph; do not instantiate or train."""
    import torch
    dataset_rows, model_rows = [], []
    for path in sorted(run.glob('*/graph.pt')):
        checkpoints = sorted(path.parent.glob('*/split_*/seed_*/checkpoint.pt'))
        if not checkpoints:
            continue
        graph = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        labels, edges = graph['y'].reshape(-1), graph['edge_index']
        matched = 0
        for chunk in edges.split(1_000_000, dim=1):
            matched += int(labels[chunk[0]].eq(labels[chunk[1]]).sum())
        dataset_rows.append({'dataset': path.parent.name, 'nodes': len(labels),
                             'undirected_edge_pairs': edges.shape[1] // 2,
                             'classes': int(labels.max()) + 1,
                             'edge_homophily': matched / edges.shape[1] if edges.shape[1] else np.nan})
        seen = set()
        for checkpoint in checkpoints:
            backbone = checkpoint.parents[2].name
            if backbone in seen:
                continue
            seen.add(backbone)
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            parameters = sum(value.numel() for name, value in saved['state_dict'].items()
                             if not name.endswith(('running_mean', 'running_var', 'num_batches_tracked')))
            recipe = saved['recipe']
            dataset_rows[-1]['features'] = saved['in_features']
            model_rows.append({'dataset': path.parent.name, 'backbone': backbone,
                               'features': saved['in_features'], 'classes': saved['out_features'],
                               'parameters': parameters, 'layers': recipe['layers'], 'width': recipe['hidden'],
                               'dropout': recipe['dropout']})
        del graph, labels, edges
    return pd.DataFrame(dataset_rows), pd.DataFrame(model_rows)


def runtime_table(frame):
    """Fixed-depth call/derived step times and the paired PtS-to-APPNP ratio."""
    frame = frame[frame['mode'].eq('fixed_100') & frame.method.isin(['appnp', 'ppr', 'pts'])]
    if frame.empty:
        return pd.DataFrame()
    keys = UNIT + ['device', 'hardware', 'threads', 'dtype']
    wide = frame.pivot(index=keys, columns='method', values=['ms', 'ms_per_step'])
    wide.columns = [f'{method}_{metric}' for metric, method in wide.columns]
    if {'pts_ms_per_step', 'appnp_ms_per_step'} <= set(wide):
        wide['pts_over_appnp_per_step'] = wide.pts_ms_per_step / wide.appnp_ms_per_step
    return wide.reset_index()


def search_table(config):
    search = config['search']
    return pd.DataFrame([{'method': method,
                          'access': 'label_aware' if method in ('cs', 'cs_pts') else 'label_free',
                          'evaluations': 26 if method == 'graph_tv' else search['trials'],
                          'settings': json.dumps(params, sort_keys=True),
                          'search': json.dumps(search.get('graph_tv') if method == 'graph_tv' else search, sort_keys=True)}
                         for method, params in config['methods'].items()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(args.config.read_text()) if args.config else None
    build(args.run, args.output, config=config, require_complete=args.require_complete, plots=not args.no_plots)


if __name__ == '__main__':
    main()
