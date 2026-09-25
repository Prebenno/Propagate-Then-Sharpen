"""Scientific plots from the minimal CSV tables; no manuscript or parent package required."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {'appnp': '#356b99', 'ppr': '#b08b2e', 'pts': '#a63859',
          'anchor': '#777777', 'logit_sharp': '#59616b'}
LABELS = {'appnp': 'APPNP', 'ppr': 'PPR-Prob', 'pts': 'PtS',
          'anchor': 'Frozen Q', 'logit_sharp': 'Logit-Sharp'}
MAIN = ('wikics', 'cora-tag', 'pubmed-tag', 'tape-arxiv23', 'ogbn-arxiv',
        'ogbn-products', 'ele-photo', 'ele-computers', 'books-history')


def analytical_trace(max_depth=100):
    """Exact two matched nine-node cliques; this is an illustration, not fitted evidence."""
    import torch
    from .methods.pts import graph_operator, normalize, sharpen
    edges = [(start + i, start + j) for start in (0, 9) for i in range(9) for j in range(i + 1, 9)]
    edges += [(i, i + 9) for i in range(9)]
    edge_index = torch.tensor(edges).T.contiguous()
    q = torch.tensor([[.9, .1]] * 9 + [[.4, .6]] * 9, dtype=torch.float64)
    operator = graph_operator(edge_index, len(q), dtype=q.dtype)
    logits, states, rows = q.log(), {}, []
    states.update(appnp=logits, ppr=q, pts=q)
    for depth in range(max_depth + 1):
        for method, state in states.items():
            p = state.softmax(1) if method == 'appnp' else normalize(state)
            for group, nodes, truth in [('A', slice(0, 9), 0), ('B', slice(9, 18), 1)]:
                rows.append({'K': depth, 'method': method, 'group': group,
                             'correct_class_probability': float(p[nodes, truth].mean()),
                             'accuracy': float(p[nodes].argmax(1).eq(truth).double().mean())})
        if depth < max_depth:
            for method, state in list(states.items()):
                source = logits if method == 'appnp' else q
                transported = .1 * source + .9 * torch.sparse.mm(operator, state)
                states[method] = sharpen(transported, 4.) if method == 'pts' else transported
    return pd.DataFrame(rows), edge_index


def build(output, partial=False):
    """Generate available empirical figures plus the deterministic analytical example."""
    output = Path(output)
    target = output / 'figures'
    target.mkdir(parents=True, exist_ok=True)
    title_suffix = ' · partial/demo' if partial else ''
    generated = []

    def table(name):
        path = output / 'tables' / (name + '.csv')
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def macro(frame):
        return frame[frame.dataset.isin(['main_mean', 'partial_mean']) & frame.backbone.eq('mlp')]

    def save(fig, name, data):
        fig.savefig(target / f'{name}.png', dpi=180, bbox_inches='tight')
        fig.savefig(target / f'{name}.pdf', bbox_inches='tight',
                    metadata={'Creator': 'PtS minimal reporting', 'CreationDate': None, 'ModDate': None})
        data.to_csv(target / f'{name}.csv', index=False)
        plt.close(fig)
        generated.append(name)

    with plt.rc_context({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'axes.axisbelow': True}):
        ladder = table('table2.label_free')
        if not ladder.empty:
            ladder = macro(ladder)
            fig, ax = plt.subplots(figsize=(6.3, 4), layout='constrained')
            for contrast, label, color in [('ppr - appnp', 'PPR-Prob − APPNP', COLORS['ppr']),
                                           ('pts - ppr', 'PtS − PPR-Prob', COLORS['pts']),
                                           ('pts - appnp', 'PtS − APPNP', '#252a30')]:
                curve = ladder[ladder.method.eq(contrast)].sort_values('sigma')
                ax.plot(curve.sigma, 100 * curve['mean'], '-o', label=label, color=color)
            ax.axhline(0, color='#999999', linewidth=.7)
            ax.set(xlabel='Gaussian severity σ', ylabel='Mean accuracy gain (pp)',
                   title='Severity-matched validation selection' + title_suffix)
            ax.grid(color='#e4e4e4'); ax.legend(frameon=False)
            save(fig, 'severity_gains', ladder)

        depth = table('depth.label_free.main_mean')
        if not depth.empty:
            depth = macro(depth)
            depth = depth[depth.metric.eq('accuracy')]
            fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained', sharex=True, sharey=True)
            variants = [('appnp', 0, 's', '-'), ('ppr', 0, 'D', '-'), ('logit_sharp', 16, 'v', '-'),
                        ('pts', 16, 'o', '-'), ('pts', 200, 'o', '--')]
            for row, sigma in enumerate([0, 2]):
                for col, alpha in enumerate([0, .1]):
                    ax = axes[row, col]
                    for method, eta, marker, linestyle in variants:
                        curve = depth[depth.sigma.eq(sigma) & depth.alpha.eq(alpha) &
                                      depth.method.eq(method) & depth.eta.eq(eta)].sort_values('K')
                        if curve.empty:
                            continue
                        label = LABELS[method] + (f' (η={eta})' if eta else '')
                        ax.plot(curve.K, 100 * curve['mean'], marker=marker, linestyle=linestyle,
                                markersize=3, color=COLORS[method], label=label)
                    ax.set(xscale='log', xlabel='Propagation depth K', ylabel='Test accuracy (%)',
                           title=f'σ={sigma}, restart α={alpha:g}')
                    ax.grid(color='#e4e4e4')
                    ax.set_xticks([1, 2, 3, 5, 10, 20, 40, 100], labels=[1, 2, 3, 5, 10, 20, 40, 100])
            axes[0, 0].legend(frameon=False, fontsize=7)
            fig.suptitle('Fixed-depth comparison on clean and severe inputs' + title_suffix)
            save(fig, 'depth_clean_severe', depth)

        transfer = table('transfer.differences.label_free')
        if not transfer.empty:
            transfer = macro(transfer)
            fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
            for ax, contrast in zip(axes, ['pts - appnp', 'pts - ppr']):
                part = transfer[transfer.method.eq(contrast)]
                if part.empty:
                    ax.set_visible(False); continue
                grid = part.pivot(index='sigma_selected', columns='sigma', values='mean') * 100
                limit = max(float(np.nanmax(np.abs(grid.to_numpy()))), .01)
                im = ax.imshow(grid, cmap='RdBu_r', vmin=-limit, vmax=limit)
                for i in range(len(grid)):
                    for j in range(len(grid.columns)):
                        v = grid.iloc[i, j]
                        if np.isfinite(v):
                            ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=8,
                                    color='white' if abs(v) > .6 * limit else '#222222')
                ax.set_xticks(range(len(grid.columns)), [f'{v:g}' for v in grid.columns])
                ax.set_yticks(range(len(grid)), [f'{v:g}' for v in grid.index])
                ax.set(xlabel='Evaluation severity σ', ylabel='Selection severity σ', title=contrast)
                fig.colorbar(im, ax=ax, shrink=.8, label='Accuracy gain (pp)')
            fig.suptitle('Severity transfer (matched draw where both are noisy)' + title_suffix)
            save(fig, 'severity_transfer', transfer)

        energy = table('energy.label_free')
        if not energy.empty:
            energy = energy[energy.dataset.eq('wikics') & energy.backbone.eq('mlp')].copy()
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), layout='constrained')
            masses = energy[energy.metric.eq('kernel_mass')][['method', 'K', 'phase', 'mean']].rename(columns={'mean': 'mass'})
            energy = energy.merge(masses, on=['method', 'K', 'phase'], validate='many_to_one')
            energy['plotted_value'] = np.where(energy.metric.eq('val_accuracy'), energy['mean'] * 100,
                                               energy['mean'] / energy.mass)
            for ax, metric, label in zip(axes, ['dirichlet', 'gini', 'val_accuracy'],
                                         ['Disagreement / kernel mass', 'Indecision / kernel mass', 'Validation accuracy (%)']):
                for method, phases in [('ppr', ['initial', 'reaction']), ('pts', ['initial', 'reaction']), ('pts', ['transport'])]:
                    part = energy[energy.metric.eq(metric) & energy.method.eq(method) & energy.phase.isin(phases)].sort_values('K')
                    ax.plot(part.K, part.plotted_value, '--s' if phases == ['transport'] else '-o',
                            markersize=3, label=LABELS[method] + (' after transport' if phases == ['transport'] else ''),
                            color='#5b6ee1' if phases == ['transport'] else COLORS[method])
                ax.set(xlabel='Propagation depth K', ylabel=label); ax.grid(axis='y', color='#e4e4e4')
            axes[0].legend(frameon=False, fontsize=7)
            fig.suptitle('WikiCS, σ=2, α=0.1, PtS η=16' + title_suffix)
            save(fig, 'energy_half_steps', energy)

        calibration = table('calibration.label_free.main_mean')
        if not calibration.empty:
            calibration = macro(calibration)
            fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout='constrained')
            for row, sigma in enumerate([0, 2]):
                for col, metric in enumerate(['nll', 'ece']):
                    ax = axes[row, col]
                    for offset, prefix, label in [(-.18, 'raw_', 'Raw'), (.18, 'cal_', 'Temperature scaled')]:
                        part = calibration[calibration.sigma.eq(sigma) & calibration.metric.eq(prefix + metric)].set_index('method')
                        methods = ['anchor', 'appnp', 'pts']
                        values = [part.loc[m, 'mean'] if m in part.index else np.nan for m in methods]
                        ax.bar(np.arange(3) + offset, values, width=.36, label=label)
                    ax.set_xticks(range(3), ['Frozen Q', 'APPNP', 'PtS'])
                    ax.set(title=f'σ={sigma}', ylabel=metric.upper())
            axes[0, 0].legend(frameon=False)
            fig.suptitle('Validation-fitted temperature scaling' + title_suffix)
            save(fig, 'calibration', calibration)

        per_node = table('per_node.differences.label_free')
        if not per_node.empty:
            per_node = macro(per_node)
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout='constrained')
            for ax, variable, label in zip(axes, ['local_homophily', 'degree_quintile', 'anchor_confidence_quintile'],
                                           ['Local homophily', 'Degree quintile', 'Anchor-confidence quintile']):
                for sigma in sorted(per_node.sigma.unique()):
                    part = per_node[per_node.variable.eq(variable) & per_node.sigma.eq(sigma)].sort_values('bucket_index')
                    ax.plot(part.bucket_index, 100 * part['mean'], '-o', label=f'σ={sigma:g}')
                    ax.set_xticks(part.bucket_index, part.bucket, rotation=25)
                ax.axhline(0, linewidth=.7, color='#999999')
                ax.set(xlabel=label, ylabel='PtS − APPNP (pp)'); ax.grid(axis='y', color='#e4e4e4')
            axes[0].legend(frameon=False)
            fig.suptitle('Node-stratum paired gains' + title_suffix)
            save(fig, 'per_node', per_node)

        analytical, edges = analytical_trace()
        fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.3), layout='constrained')
        angles = np.linspace(0, 2 * np.pi, 9, endpoint=False) + np.pi / 2
        circle = .72 * np.column_stack((np.cos(angles), np.sin(angles)))
        xy = np.concatenate((circle + [-1.15, 0], circle + [1.15, 0]))
        for a, b in edges.T.tolist():
            axes[0].plot(xy[[a, b], 0], xy[[a, b], 1], color='#bbbbbb', linewidth=.5, zorder=1)
        axes[0].scatter(xy[:9, 0], xy[:9, 1], color=COLORS['appnp'], s=30, zorder=2)
        axes[0].scatter(xy[9:, 0], xy[9:, 1], color=COLORS['ppr'], marker='s', s=30, zorder=2)
        axes[0].text(-1.15, -1.15, 'Class 1\nQ=(0.9, 0.1)', ha='center')
        axes[0].text(1.15, -1.15, 'Class 2\nQ=(0.4, 0.6)', ha='center')
        axes[0].set_aspect('equal'); axes[0].axis('off')
        for method in ('appnp', 'ppr', 'pts'):
            part = analytical[analytical.method.eq(method) & analytical.group.eq('B') & analytical.K.le(10)]
            axes[1].plot(part.K, part.correct_class_probability, '-o', markersize=3,
                         label=LABELS[method], color=COLORS[method])
        axes[1].axhline(.5, color='#777777', linestyle=':', linewidth=.8)
        axes[1].set(xlabel='Propagation depth K', ylabel='Correct-class probability in group B', ylim=(0, 1))
        axes[1].legend(frameon=False, fontsize=8)
        fig.suptitle('Analytical illustration: two matched cliques, α=0.1, η=4')
        save(fig, 'two_community', analytical)
    return generated
