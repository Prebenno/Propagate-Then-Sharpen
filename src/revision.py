"""Run revision experiments on saved predictions."""

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import optuna
import torch
import yaml

from .backbones import load_checkpoint, preprocess
from .diagnostics import accuracy, metrics
from .experiments import context_info, load_context, unit_info
from .methods.pts import graph_operator, normalize, sharpen
from .revision_methods import ARCHIVE_METHOD_NAMES, hard_react, power_react, predict, tune
from .seeds import stable_seed, study_seed

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('controls', 'backbones', 'calibration', 'outage', 'depth', 'energy',
          'per_node', 'products')


DEPTHS = (1, 2, 3, 5, 10, 20, 40, 100)
ETAS = (0.5, 1, 2, 4, 8, 16, 32, 64, 128, 200)
TAUS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128)
RULES = ("appnp", "ppr", "pts", "power", "hard", "logit_sharp")


def _predict(method, params, data, operators):
    probability, _ = predict(
        method, data["q"], data["edges"], params, logits=data["logits"],
        train_index=data["splits"].get("train_index"),
        train_labels=data.get("train_labels"), operators=operators)
    return probability


def _operator(data, operators):
    if "sym" not in operators:
        q = data["q"]
        operators["sym"] = graph_operator(
            data["edges"], len(q), dtype=q.dtype, device=q.device)
    return operators["sym"]


#Fit temperature by validation NLL.
@torch.no_grad()
def fit_temperature(probability, labels, validation):
    if len(validation) == 0:
        raise ValueError("Temperature fitting requires validation nodes")
    log_p = probability[validation].clamp_min(1e-12).log()
    target = labels[validation]

    def nll(temperature):
        scaled = torch.softmax(log_p / temperature, dim=1)
        return float(-scaled.gather(1, target[:, None]).clamp_min(1e-12).log().mean())

    candidates = torch.logspace(-1.3, 1.6, 60).tolist() + [1.0]
    best = min(candidates, key=nll)
    low, high = best / 1.6, best * 1.6
    for _ in range(3):
        best = min(torch.linspace(low, high, 25).tolist(), key=nll)
        width = (high - low) / 8
        low, high = max(best - width, 1e-3), best + width
    return float(best)


#Use clean-selected settings and fit temperature on clean validation.
@torch.no_grad()
def calibration_rows(clean_data, target_data, settings, operators):
    rows = []
    for method, params in settings.items():
        clean = _predict(method, params, clean_data, operators)
        temperature = fit_temperature(
            clean, clean_data["y"], clean_data["splits"]["val_index"])
        target = clean if target_data is clean_data else _predict(
            method, params, target_data, operators)
        calibrated = torch.softmax(target.clamp_min(1e-12).log() / temperature, dim=1)
        for group, probability, value in (("raw", target, 1.0),
                                          ("calibrated", calibrated, temperature)):
            rows.append({
                "method": method, "group": group,
                "protocol": "clean_sel__clean_T" + ("__raw" if group == "raw" else ""),
                "temperature": value, "fitted_temperature": temperature,
                **metrics(probability, target_data["y"], target_data["splits"]["test_index"]),
            })
    return rows


def _section(config, name):
    return config.get(name, config)


def _strengths(method, config):
    if method == "pts":
        return sorted(set(float(x) for x in config.get("etas", ETAS)))
    if method == "power":
        return sorted(set(float(x) for x in config.get("taus", TAUS)))
    if method == "logit_sharp":
        return sorted(set(float(x) for x in config.get("logit_etas", (16, 200))))
    if method in {"appnp", "ppr"}:
        return [0.0]
    if method == "hard":
        return [1.0]
    raise ValueError(f"Unsupported trajectory method: {method}")


def _trajectory(data, operator, method, alpha, strength, steps, record_at):
    logit_space = method in {"appnp", "logit_sharp"}
    source = data["logits"] if logit_space else data["q"]
    state = source
    to_field = (lambda x: x.softmax(1)) if logit_space else normalize
    if 0 in record_at:
        yield 0, "pre", to_field(state)
    for step in range(1, steps + 1):
        state = alpha * source + (1 - alpha) * torch.sparse.mm(operator, state)
        if step in record_at:
            yield step, "transport", to_field(state)
        if method == "pts":
            state = sharpen(state, strength)
        elif method == "power":
            state = power_react(state, strength)
        elif method == "hard":
            state = hard_react(state)
        elif method == "logit_sharp" and strength > 0:
            state = state + strength * state.softmax(1)
        if step in record_at:
            yield step, "reaction", to_field(state)


#Select strength by mean validation accuracy across depths.
#Ties within 1e-12 use the weaker strength.
@torch.no_grad()
def depth_rows(data, config, operators):
    cfg = _section(config, "depth")
    steps = sorted(set(int(k) for k in cfg.get("steps", DEPTHS)))
    if not steps or steps[0] < 1:
        raise ValueError("Depth grid must contain positive steps")
    if len(data["splits"]["val_index"]) == 0:
        raise ValueError("Depth selection requires validation nodes")
    operator = _operator(data, operators)
    rows = []
    for alpha in cfg.get("alphas", (0.0, 0.1)):
        for method in cfg.get("methods", RULES):
            curves = []
            for strength in _strengths(method, cfg):
                curve = []
                for step, phase, probability in _trajectory(
                        data, operator, method, float(alpha), strength, max(steps), set(steps)):
                    if phase == "reaction":
                        curve.append({
                            "method": method, "group": "validation_frozen", "alpha": float(alpha),
                            "strength": strength, "K": step,
                            "accuracy": accuracy(probability, data["y"], data["splits"]["test_index"]),
                            "val_accuracy": accuracy(probability, data["y"], data["splits"]["val_index"]),
                        })
                curves.append((sum(row["val_accuracy"] for row in curve) / len(curve), curve))
            if not curves:
                raise ValueError(f"Empty strength grid for {method}")
            best = max(value for value, _ in curves)
            chosen_score, chosen = next((value, curve) for value, curve in curves if value >= best - 1e-12)
            rows.extend(dict(row, selection_val_accuracy=chosen_score,
                             selection_metric="mean_validation_accuracy_over_depths") for row in chosen)
    return rows


#Symmetric-kernel interaction traces, excluding the anchor term.
@torch.no_grad()
def energy_rows(data, config, operators):
    cfg = _section(config, "energy")
    steps = int(cfg.get("steps", 100))
    if steps < 1:
        raise ValueError("Energy trajectories require positive steps")
    record_at = {int(k) for k in cfg.get("record_steps", list(range(21)) + [100])
                 if 0 <= int(k) <= steps}
    record_at.update(range(min(steps, 20) + 1))
    record_at.add(steps)
    operator = _operator(data, operators)
    q = data["q"]
    row_sum = torch.sparse.mm(operator, torch.ones_like(q[:, :1])).squeeze(1)
    kernel_mass = float(row_sum.sum())
    rows = []
    for alpha in cfg.get("alphas", (0.0, 0.1)):
        for method in cfg.get("methods", RULES):
            for strength in _strengths(method, cfg):
                for step, phase, probability in _trajectory(
                        data, operator, method, float(alpha), strength, steps, record_at):
                    agreement = float((probability * torch.sparse.mm(operator, probability)).sum())
                    weighted_square = float((row_sum[:, None] * probability.square()).sum())
                    dirichlet = 0.5 * (weighted_square - agreement)
                    gini = 0.5 * (kernel_mass - weighted_square)
                    rows.append({
                        "method": method, "group": "fixed_grid", "alpha": float(alpha),
                        "strength": strength, "K": step, "phase": phase,
                        "dirichlet": dirichlet, "gini": gini, "potts": dirichlet + gini,
                        "kernel_mass": kernel_mass,
                        "accuracy": accuracy(probability, data["y"], data["splits"]["test_index"]),
                        "val_accuracy": accuracy(probability, data["y"], data["splits"]["val_index"]),
                    })
    return rows


def _quintiles(values):
    rank = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return np.minimum(rank * 5 // max(len(values), 1), 4)


#Retrospective homophily groups and paired correction/damage counts.
#True labels define groups, not predictions. Empty groups have null rates.
#Confidence uses the saved float32 precision.
@torch.no_grad()
def per_node_rows(data, settings, operators, clean_data=None):
    q = data["q"]
    n_nodes = len(q)
    labels = data["y"].detach().cpu().numpy()
    test = data["splits"]["test_index"].detach().cpu().numpy()
    n_test = len(test)
    if n_test == 0:
        raise ValueError("Per-node diagnostics require test nodes")
    # Exclude self-loops from homophily and degree.
    edges = _operator(data, operators).coalesce().indices().detach().cpu().numpy()
    source, target = edges[:, edges[0] != edges[1]]
    degree = np.bincount(source, minlength=n_nodes)
    same = np.bincount(source, weights=labels[source] == labels[target], minlength=n_nodes)
    homophily = same / np.maximum(degree, 1)
    isolated = degree[test] == 0
    hom_bins = np.clip(np.searchsorted(
        [-1e-9, .2, .4, .6, .8, 1.0 + 1e-9], homophily[test], side="right") - 1, 0, 4)
    hom_bins[isolated] = 5
    degree_q = _quintiles(degree[test])
    confidence = q.max(1).values.detach().cpu().numpy().astype(np.float32)[test]
    confidence_q = _quintiles(confidence)
    predictions = {"anchor": q.argmax(1).detach().cpu().numpy()}
    for method, params in settings.items():
        if method != "anchor":
            predictions[method] = _predict(method, params, data, operators).argmax(1).detach().cpu().numpy()
    ok = {method: values[test] == labels[test] for method, values in predictions.items()}
    anchor_ok = ok["anchor"]
    groups = [("context", "all_test", np.ones(n_test, dtype=bool)),
              ("priors", "anchor_wrong_all", ~anchor_ok),
              ("priors", "anchor_right_all", anchor_ok),
              ("priors", "confident_wrong", (confidence >= .9) & ~anchor_ok),
              ("priors", "confident_right", (confidence >= .9) & anchor_ok)]
    if clean_data is not None:
        clean_ok = clean_data["q"].argmax(1).detach().cpu().numpy()[test] == labels[test]
        groups += [("corruption", "corruption_induced_error", clean_ok & ~anchor_ok),
                   ("corruption", "pre_existing_error", ~clean_ok & ~anchor_ok),
                   ("corruption", "still_correct", clean_ok & anchor_ok),
                   ("corruption", "corruption_fixed", ~clean_ok & anchor_ok)]
    for bucket, label in enumerate(("0-.2", ".2-.4", ".4-.6", ".6-.8", ".8-1", "isolated")):
        groups.append(("homophily", label, hom_bins == bucket))
    for bucket in range(5):
        groups += [("quintiles", f"degree_Q{bucket + 1}", degree_q == bucket),
                   ("quintiles", f"anchor_confidence_Q{bucket + 1}", confidence_q == bucket)]
    low_h = (~isolated) & (homophily[test] < .4)
    groups += [("joint", "homophily_lt_0.4_all", low_h),
               ("joint", "homophily_lt_0.4_correct", low_h & anchor_ok),
               ("joint", "homophily_lt_0.4_wrong", low_h & ~anchor_ok)]
    if "lame" in ok:
        groups += [("lame_correctness", "lame_wrong", ~ok["lame"]),
                   ("lame_correctness", "lame_right", ok["lame"])]

    rows = []
    for table, group, mask in groups:
        count = int(mask.sum())
        for method, correct in ok.items():
            number_correct = int((correct & mask).sum())
            rows.append({"method": method, "table": table, "group": group,
                         "n": count, "n_test": n_test, "fraction_test": count / n_test,
                         "n_correct": number_correct,
                         "accuracy": number_correct / count if count else None})
        if "pts" in ok:
            for comparator in sorted(ok.keys() - {"pts"}):
                corrected = int((~ok[comparator] & ok["pts"] & mask).sum())
                damaged = int((ok[comparator] & ~ok["pts"] & mask).sum())
                errors = int((~ok[comparator] & mask).sum())
                correct = count - errors
                rows.append({
                    "method": "pts", "comparator": comparator, "table": "transitions",
                    "stratum_table": table, "group": group, "n": count, "n_test": n_test,
                    "fraction_test": count / n_test, "corrected": corrected, "damaged": damaged,
                    "comparator_wrong": errors, "comparator_correct": correct,
                    "correction_fraction": corrected / n_test, "damage_fraction": damaged / n_test,
                    "net_gain": (corrected - damaged) / n_test,
                    "correction_rate": corrected / errors if errors else None,
                    "damage_rate": damaged / correct if correct else None,
                    "within_group_gain": (corrected - damaged) / count if count else None,
                })
    return rows


def load_config(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else set(seen)
    if path in seen:
        raise ValueError('Cyclic configuration inheritance')
    seen.add(path)
    overlay = yaml.safe_load(path.read_text(encoding='utf-8'))
    base = overlay.pop('extends', None)
    config = load_config(path.parent / base, seen) if base else {}

    def merge(left, right):
        for key, value in right.items():
            if isinstance(value, dict) and isinstance(left.get(key), dict):
                merge(left[key], value)
            else:
                left[key] = copy.deepcopy(value)
    merge(config, overlay)
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_rows(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.csv.tmp')
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True)
                          if isinstance(value, (list, dict)) else value
                          for key, value in row.items()} for row in rows)
    temporary.replace(path)


def evaluate(method, params, data, operators):
    return predict(method, data['q'], data['edges'], params,
                   logits=data['logits'], train_index=data['splits']['train_index'],
                   train_labels=data['train_labels'], operators=operators)


#Tune each method on validation nodes.
def select(context, data, info, methods, config, operators, destination):
    search = copy.deepcopy(config['search'])
    revision = config['revision']
    if info['dataset'] == 'ogbn-products':
        search['max_steps'] = min(search['max_steps'], revision['max_steps_products'])
    sigma, draw = context_info(context)
    if sigma == 0:
        draw = -1
    tag = revision.get('optuna_seed_tag', config['optuna_seed_tag'])
    if info['backbone'] != 'mlp':
        tag += '_' + info['backbone']
    path = destination / (context.stem + '.selections.json')
    saved = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    saved = saved.get('selections', saved)
    validation = data['splits']['val_index']
    for method in methods:
        if method in saved:
            continue
        original = revision.get('optuna_method_names', ARCHIVE_METHOD_NAMES).get(method, method)
        search['seed'] = study_seed(info['dataset'], info['split'], info['seed'],
                                    sigma, draw, original, tag=tag)
        parameters, probability, _, history = tune(
            method, lambda name, params: evaluate(name, params, data, operators),
            config['methods'][method], validation, data['y'][validation], search)
        saved[method] = {
            'parameters': parameters,
            'validation_accuracy': accuracy(probability, data['y'], validation),
            'search': history,
        }
        write_json(path, saved)
        print(f'  selected {context.stem} {method}', flush=True)
    return {method: saved[method]['parameters'] for method in methods}


def control_rows(data, settings, operators):
    rows = []
    for method, params in {'anchor': {}, **settings}.items():
        probability, _ = evaluate(method, params, data, operators)
        rows.append({'method': method, 'parameters': params,
                     'validation_accuracy': accuracy(probability, data['y'], data['splits']['val_index']),
                     **metrics(probability, data['y'], data['splits']['test_index'])})
        if method == 'pts':
            off = dict(params, eta=0.0)
            probability, _ = evaluate('pts', off, data, operators)
            rows.append({'method': 'reaction_off', 'parameters': off,
                         'selection': 'PtS parameters with reaction disabled; not independently tuned',
                         **metrics(probability, data['y'], data['splits']['test_index'])})
    return rows


def outage_mask(tag, dataset, split, draw, num_nodes):
    generator = torch.Generator(device='cpu').manual_seed(
        stable_seed(tag, dataset, split, 'outage', 'test', draw, bits=63))
    return torch.rand(num_nodes, generator=generator, dtype=torch.float32)


#Replace whole feature rows with training means.
@torch.no_grad()
def outage_rows(folder, graph, split, clean_data, settings, config, device, data_root, operators,
                draws):
    from .data import load_dataset
    info = unit_info(folder)
    dataset = load_dataset(info['dataset'], data_root, splits=info['split'] + 1)
    if not torch.equal(dataset['y'], graph['y']) or not torch.equal(dataset['edge_index'], graph['edge_index']):
        raise ValueError('Outage feature data do not match the saved graph/node order')
    checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
    x = preprocess(dataset['x'], split['train_index'], checkpoint['preprocessing'])
    mean = x[split['train_index']].mean(0, keepdim=True)
    model = load_checkpoint(folder / 'checkpoint.pt', graph['edge_index'], device)
    clean_logits = model(x.to(device)).to(dtype=clean_data['q'].dtype)
    clean_q = clean_logits.softmax(1)
    tolerance = 1e-3 if clean_q.dtype == torch.float32 else 1e-4
    if not torch.allclose(clean_q, clean_data['q'], atol=tolerance, rtol=tolerance):
        raise ValueError('Outage checkpoint/feature replay does not match saved clean predictions')
    cfg = config['outage']
    rows = []
    contexts = [(-1, 0.0, torch.zeros(len(x), dtype=torch.bool))]
    for draw in draws:
        uniform = outage_mask(cfg['tag'], info['dataset'], info['split'], draw, len(x))
        contexts.extend((draw, float(rate), uniform < rate) for rate in cfg['rates'])
    for draw, rate, failed in contexts:
        logits = (clean_logits if rate == 0 else
                  model(torch.where(failed[:, None], mean, x).to(device)).to(dtype=clean_q.dtype))
        data = dict(clean_data, logits=logits, q=logits.softmax(1))
        for row in control_rows(data, settings, operators):
            if row['method'] != 'reaction_off':
                rows.append({**row, 'draw': draw, 'rho': rate, 'treatment': 'mean',
                             'subset': 'all_test', 'n': len(data['splits']['test_index']),
                             'failed_node_fraction': float(failed.double().mean()),
                             'selection_sigma': 0.0})
    return rows


#Validation and test-label grid diagnostics.
@torch.no_grad()
def products_rows(data, config, operators):
    cfg = config.get('products', {})
    alphas = cfg.get('alphas', [0, .0001, .0003, .001, .003, .01, .02, .03, .05, .075,
                              .1, .15, .2, .25, .3, .4, .5, .6, .7, .8, .9, 1])
    cap = min(config['search']['max_steps'], config['revision']['max_steps_products'])
    depths = cfg.get('steps', list(range(1, cap + 1)))
    operator = _operator(data, operators)
    rows = []
    for method in ('appnp', 'ppr'):
        best = {'validation': None, 'test': None}
        for alpha in alphas:
            for steps, phase, probability in _trajectory(
                    data, operator, method, float(alpha), 0.0, max(depths), set(depths)):
                if phase != 'reaction':
                    continue
                params = dict(alpha=float(alpha), steps=int(steps))
                values = {'validation': accuracy(probability, data['y'], data['splits']['val_index']),
                          'test': accuracy(probability, data['y'], data['splits']['test_index'])}
                for selector in best:
                    if best[selector] is None or values[selector] > best[selector]['selection_accuracy']:
                        best[selector] = dict(method=method, parameters=params,
                                              selection_accuracy=values[selector],
                                              accuracy=values['test'], validation_accuracy=values['validation'])
        for selector, row in best.items():
            rows.append(dict(row, selector=selector,
                             protocol='finite_grid_diagnostic_never_deployed' if selector == 'test'
                             else 'finite_grid_validation'))
    return rows


def run(args):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    torch.set_num_threads(args.threads)
    config = load_config(args.config)
    if args.trials is not None:
        config['search']['trials'] = args.trials
    stages = list(STAGES) if 'all' in args.experiment else args.experiment
    run_root = args.run.resolve()
    output = (args.output or run_root / 'revision').resolve()
    if output == run_root or output.is_relative_to(ROOT / 'results'):
        raise ValueError('Use a separate generated output directory, not --run or published results/')
    names = config['datasets']['main'] if not args.datasets or 'all' in args.datasets else args.datasets
    units = [folder for folder in sorted(run_root.glob('*/*/split_*/seed_*'))
             if (folder / 'checkpoint.pt').is_file()]
    units = [folder for folder in units if
             unit_info(folder)['dataset'] in names and
             (not args.backbones or unit_info(folder)['backbone'] in args.backbones) and
             (args.seeds is None or unit_info(folder)['seed'] in args.seeds) and
             (args.splits is None or unit_info(folder)['split'] < args.splits)]
    if not units:
        raise FileNotFoundError('No prepared units match. Run train.py/main.py to create checkpoints and Q/Z first.')
    output.mkdir(parents=True, exist_ok=True)
    sigmas = args.sigmas if args.sigmas is not None else config['revision']['sigmas']
    draws = args.draws if args.draws is not None else list(range(config['units']['draws']))
    processed = 0
    for folder in units:
        info = unit_info(folder)
        wanted = [s for s in stages if
                  (s == 'backbones' and info['backbone'] != 'mlp') or
                  (s != 'backbones' and info['backbone'] == 'mlp' and
                   (s != 'products' or info['dataset'] == 'ogbn-products'))]
        if not wanted:
            continue
        unit_sigmas = [0.0] if wanted == ['outage'] else sigmas
        paths = sorted(folder.glob('s*_d*.pt'), key=context_info)
        paths = [p for p in paths if context_info(p)[0] in unit_sigmas and
                 (context_info(p)[0] == 0 or context_info(p)[1] in draws)]
        clean_paths = [p for p in folder.glob('s*_d*.pt') if context_info(p)[0] == 0]
        clean_path = next((p for p in clean_paths if context_info(p)[1] == -1),
                          clean_paths[0] if clean_paths else None)
        # Some saved clean contexts use s0_d0.
        paths = [p for p in paths if context_info(p)[0] != 0 or p == clean_path]
        expected = [(float(sigma), draw) for sigma in unit_sigmas if sigma != 0 for draw in draws]
        missing = set(expected) - {context_info(p) for p in paths}
        if missing or (0 in unit_sigmas and clean_path is None):
            raise FileNotFoundError(f'{folder}: missing requested Q/Z contexts {sorted(missing)}; '
                                    'prepare them using train.py --sigmas/--draws first')
        if not paths:
            raise FileNotFoundError(f'{folder}: no requested prediction contexts')
        destination = output / info['dataset'] / info['backbone'] / folder.parent.name / folder.name
        destination.mkdir(parents=True, exist_ok=True)
        graph = torch.load(folder.parents[2] / 'graph.pt', map_location='cpu', weights_only=True)
        split = torch.load(folder / 'split.pt', map_location='cpu', weights_only=True)
        operators = {}
        clean_data = load_context(clean_path, graph, split, args.device) if clean_path else None
        methods = config['revision']['methods'] if info['backbone'] == 'mlp' else ['appnp', 'ppr', 'pts']
        need_matched = bool(set(wanted) & {'controls', 'backbones', 'per_node', 'products'})
        clean_methods = sorted(set(
            (config['calibration']['methods'] if 'calibration' in wanted else []) +
            (config['outage']['methods'] if 'outage' in wanted else [])))
        if clean_methods and clean_data is None:
            raise FileNotFoundError(f'{folder}: clean predictions required for clean-selected protocols')
        roster = list(dict.fromkeys((methods if need_matched else []) + clean_methods))
        clean_settings = select(clean_path, clean_data, info, roster, config, operators, destination) if clean_methods else {}
        records = {stage: [] for stage in wanted}
        print(f"{info['dataset']}/{info['backbone']}/{folder.parent.name}/{folder.name}", flush=True)
        for context in paths:
            sigma, draw = context_info(context)
            if sigma == 0:
                draw = -1
            data = clean_data if context == clean_path else load_context(context, graph, split, args.device)
            metadata = dict(info, sigma=sigma, draw=draw)
            settings = select(context, data, info, roster, config, operators, destination) if need_matched else {}
            for stage in wanted:
                if stage in ('controls', 'backbones'):
                    rows = control_rows(data, {m: settings[m] for m in methods}, operators)
                elif stage == 'calibration':
                    rows = calibration_rows(clean_data, data,
                                            {m: clean_settings[m] for m in config['calibration']['methods']}, operators)
                elif stage == 'depth':
                    rows = depth_rows(data, config, operators)
                elif stage == 'energy':
                    rows = energy_rows(data, config, operators)
                elif stage == 'per_node':
                    rows = per_node_rows(data, {m: settings[m] for m in config['per_node']['methods']},
                                         operators, clean_data=clean_data)
                elif stage == 'products' and sigma == 2:
                    rows = products_rows(data, config, operators)
                    # Compare clean and severity-matched selection.
                    if clean_data is None:
                        raise FileNotFoundError('Products selection audit requires clean predictions')
                    if not clean_settings:
                        clean_settings = select(clean_path, clean_data, info, roster, config, operators, destination)
                    rows += [dict(row, protocol='clean_selected') for row in control_rows(
                        data, {m: clean_settings[m] for m in ('appnp', 'ppr', 'pts')}, operators)]
                else:
                    continue
                records[stage].extend({**metadata, **row} for row in rows)
        if 'outage' in wanted:
            records['outage'] = [{**info, **row} for row in outage_rows(
                folder, graph, split, clean_data,
                {m: clean_settings[m] for m in config['outage']['methods']},
                config, args.device, args.data_root, operators, draws)]
        for stage, rows in records.items():
            write_rows(destination / f'{stage}.csv', rows)
        processed += 1
    if not processed:
        raise ValueError('Selected experiments do not apply to the requested backbones/datasets')
    print(f'Completed {processed} units; new outputs: {output}', flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--run', type=Path, required=True)
    result.add_argument('--output', type=Path)
    result.add_argument('--config', type=Path, default=ROOT / 'configs/revision.yaml')
    result.add_argument('--experiment', nargs='+', choices=['all', *STAGES], default=['all'])
    result.add_argument('--datasets', nargs='+')
    result.add_argument('--backbones', nargs='+', choices=['mlp', 'gcn', 'sage'])
    result.add_argument('--seeds', nargs='+', type=int)
    result.add_argument('--splits', type=int)
    result.add_argument('--sigmas', nargs='+', type=float)
    result.add_argument('--draws', nargs='+', type=int)
    result.add_argument('--trials', type=int)
    result.add_argument('--data-root', type=Path, default=Path('data'))
    result.add_argument('--device', default='cpu')
    result.add_argument('--threads', type=int, default=8)
    return result


def main():
    args = parser().parse_args()
    if args.threads < 1 or (args.splits is not None and args.splits < 1):
        raise ValueError('Threads and split count must be positive')
    run(args)


if __name__ == '__main__':
    main()
