"""
Aggregate experiment results into CSV tables, used for generating tables and everything needed for the paper
"""

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
    'timing': ['method'],
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
    'per_node': ['accuracy'], 'timing': ['ms'],
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
                     'dataset': 'main_mean', 'mean': group['mean'].mean(), 'sd': np.nan,
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
    together = a.merge(b, on=keys, suffixes=('_left', '_right'))
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
    rows = []
    for path in sorted(Path(run).glob('*/*/split_*/seed_*/main.selections.json')):
        folder = path.parent
        dataset, backbone = folder.parents[2].name, folder.parents[1].name
        for context, methods in json.loads(path.read_text()).items():
            sigma = float(context.split('_d')[0][1:])
            if sigma != 2:
                continue
            for method, record in methods.items():
                access = 'label_aware' if method in ('cs', 'cs_pts') else 'label_free'
                for parameter, value in record['parameters'].items():
                    if isinstance(value, (float, int)):
                        rows.append({'dataset': dataset, 'backbone': backbone, 'sigma': sigma,
                                     'method': method, 'access': access, 'parameter': parameter,
                                     'value': value})
    if not rows:
        return pd.DataFrame()
    keys = ['dataset', 'backbone', 'sigma', 'method', 'access', 'parameter']
    return pd.DataFrame(rows).groupby(keys, as_index=False).agg(
        median=('value', 'median'), n_contexts=('value', 'size'))


def save_table(frame, directory, name):
    if not frame.empty:
        frame.to_csv(directory / f'{name}.csv', index=False)


def build(run, output=None):
    run = Path(run)
    output = Path(output) if output is not None else run / 'report'
    tables = output / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    frames = read_results(run)
    for (stage, access), frame in frames.items():
        summary = reduce(frame, AXES[stage], METRICS[stage])
        save_table(summary, tables, f'{stage}.{access}')
        roster = (MAIN[:6] if stage == 'calibration' else ('wikics',) if stage == 'energy'
                  else tuple(name for name in MAIN if name != 'ogbn-products') if stage == 'per_node'
                  else MAIN)
        save_table(macro(summary, roster), tables, f'{stage}.{access}.main_mean')
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
    print(f'Tables: {tables}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    build(args.run, args.output)


if __name__ == '__main__':
    main()
