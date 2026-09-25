"""Optuna and graphtv hyperparameter tuning"""

import optuna


def accuracy(q, index, labels):
    return float(q[index].argmax(1).eq(labels).double().mean())


def _suggest(trial, method, base, search):
    params = dict(base)
    if method in {"cs", "cs_pts"}:
        params.update(alpha_correct=trial.suggest_float("alpha_correct", 0, 1),
                      alpha_smooth=trial.suggest_float("alpha_smooth", 0, 1))
    else:
        if method != "lame":
            params["alpha"] = trial.suggest_float("alpha", 0, 1)
        params["steps"] = trial.suggest_int("K", 1, search["max_steps"])
    if method in {"pts", "pts_rn", "cs_pts", "lame"}:
        key = "strength" if method == "lame" else "eta"
        switch = "use_graph" if method == "lame" else "use_reaction"
        if trial.suggest_categorical(switch, [0, 1]):
            params[key] = 10 ** trial.suggest_float("log10_" + key, *search["eta_log10"])
        else:
            params[key] = 0.
    return params


#Pick parameters using validation accuracy.
def tune(method, predict, base, val_index, val_labels, search):
    if method == "graph_tv":
        return _tv_path(predict, base, val_index, val_labels, search["graph_tv"])
    if search["trials"] < 1 or search["max_steps"] < 1:
        raise ValueError("Search trial count and maximum depth must be positive")
    seed = search.get("seed", 1)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    history, best = [], None

    def objective(trial):
        nonlocal best
        params = _suggest(trial, method, base, search)
        q, diagnostics = predict(method, params)
        score = accuracy(q, val_index, val_labels)
        history.append({"trial": trial.number, "params": params, "validation_accuracy": score})
        if best is None or score > best[0]:
            best = score, params, q, diagnostics
        return score

    study.optimize(objective, n_trials=search["trials"], show_progress_bar=False)
    _, params, q, diagnostics = best
    return params, q, diagnostics, {"kind": "tpe", "seed": seed, "trials": history}


def _tv_path(predict, base, val_index, val_labels, search):
    low, high, count = search["log10_grid"]
    grid = [0.] + [10 ** (low + i * (high - low) / (count - 1)) for i in range(count)]
    dual, best, below = None, None, 0
    history = []
    for i, lam in enumerate(grid):
        params = dict(base, lam=lam)
        q, diagnostics, current = predict("graph_tv", dict(params, dual_init=dual, return_dual=True))
        if current is not None:
            dual = current
        score = accuracy(q, val_index, val_labels)
        eligible = diagnostics["relative_primal_dual_gap"] <= search["select_gap"]
        history.append({"trial": i, "params": params, "validation_accuracy": score,
                        "eligible": eligible, "diagnostics": diagnostics})
        if eligible and (best is None or score > best[0]):
            best = score, dict(params, warm_start_lambdas=grid[:i]), q, diagnostics
        below = below + 1 if best and score < best[0] - search["drop_pp"] / 100 else 0
        if search["consecutive"] and below >= search["consecutive"]:
            break
    if best is None:
        raise RuntimeError("No Graph-TV candidate met the selection gap")
    _, params, q, diagnostics = best
    return params, q, diagnostics, {"kind": "ascending_warm_start_grid", "trials": history,
                                     "skipped_lambdas": grid[len(history):],
                                     "select_gap": search["select_gap"]}
