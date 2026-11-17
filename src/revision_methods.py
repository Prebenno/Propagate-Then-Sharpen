"""Temperature and hard-thresholding controls."""

import optuna
import torch

from .methods.baselines import predict as original_predict
from .methods.pts import graph_operator, normalize


METHODS = ("appnp", "ppr", "pts", "power", "hard", "lame")
ARCHIVE_METHOD_NAMES = {method: ("potts" if method == "pts" else method)
                        for method in METHODS}


#Apply p**(1 + tau) and preserve row mass.
def power_react(state, tau, epsilon=1e-12):
    if tau == 0:
        return state
    if tau < 0:
        raise ValueError("Temperature-sharpening tau must be nonnegative")
    mass = state.sum(dim=1, keepdim=True)
    p = (state / mass).clamp_min(epsilon)
    p = p / p.sum(dim=1, keepdim=True)
    return mass * torch.softmax((1.0 + tau) * p.log(), dim=1)


#Preserve row mass and split it equally across tied maxima.
def hard_react(state):
    maxima = state == state.amax(dim=1, keepdim=True)
    return state.sum(dim=1, keepdim=True) * maxima / maxima.sum(dim=1, keepdim=True)


@torch.no_grad()
def _control(q, edges, *, alpha, steps, tau, reaction, operator=None, epsilon=1e-12):
    if operator is None:
        operator = graph_operator(edges, q.shape[0], dtype=q.dtype, device=q.device)
    state = q
    for _ in range(steps):
        state = alpha * q + (1 - alpha) * torch.sparse.mm(operator, state)
        if tau != 0:
            state = (power_react(state, tau, epsilon) if reaction == "power"
                     else hard_react(state))
    return normalize(state)


def predict(method, q, edges, params, **kwargs):
    if method not in {"power", "hard"}:
        return original_predict(method, q, edges, params, **kwargs)
    settings = dict(params)
    tau = settings["tau"]
    if method == "hard" and tau not in (0, 1):
        raise ValueError("Hard thresholding tau is an off/on switch: 0 or 1")
    if tau < 0:
        raise ValueError("Temperature-sharpening tau must be nonnegative")
    if tau == 0:
        # Use the PPR path for an exact off setting.
        settings.pop("tau")
        settings.pop("epsilon", None)
        return original_predict("ppr", q, edges, settings, **kwargs)
    operators = kwargs.get("operators")
    if operators is None:
        operators = {}
    if "sym" not in operators:
        operators["sym"] = graph_operator(edges, len(q), dtype=q.dtype, device=q.device)
    settings["operator"] = operators["sym"]
    return _control(q, edges, reaction=method, **settings), {}


def _suggest(trial, method, base, search):
    params = dict(base)
    params["alpha"] = trial.suggest_float("alpha", 0.0, 1.0)
    params["steps"] = trial.suggest_int("K", 1, search["max_steps"])
    on = trial.suggest_categorical("use_reaction", [0, 1])
    if method == "hard":
        params["tau"] = float(on)
    elif on:
        params["tau"] = 10 ** trial.suggest_float(
            "log10_tau", *search.get("tau_log10", [-2.0, 1.5]))
    else:
        params["tau"] = 0.0
    return params


#Tune controls on validation labels.
#Set search['seed'] with study_seed and ARCHIVE_METHOD_NAMES[method].
def tune(method, predict, base, val_index, val_labels, search):
    if method not in {"power", "hard"}:
        from .selection import tune as original_tune
        return original_tune(method, predict, base, val_index, val_labels, search)
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
        score = float(q[val_index].argmax(1).eq(val_labels).double().mean())
        history.append({"trial": trial.number, "params": params, "validation_accuracy": score})
        if best is None or score > best[0]:
            best = score, params, q, diagnostics
        return score

    study.optimize(objective, n_trials=search["trials"], show_progress_bar=False)
    _, params, q, diagnostics = best
    return params, q, diagnostics, {"kind": "tpe", "seed": seed, "trials": history}
