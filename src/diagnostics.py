"""Depth, energy, calibration, node-level and timing experiments."""

from statistics import median
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .methods.baselines import predict
from .methods.pts import graph_operator, normalize, sharpen

DEPTHS = (1, 2, 3, 5, 10, 20, 40, 100)
ALPHAS = (0.0, 0.1)


def accuracy(probability, labels, index):
    return float(probability[index].argmax(1).eq(labels[index]).double().mean())


#Compute accuracy, NLL, Brier score and ECE.
def metrics(probability, labels, index):
    p, y = probability[index].double(), labels[index]
    confidence, prediction = p.max(1)
    correct = prediction.eq(y).double()
    onehot = torch.zeros_like(p).scatter_(1, y[:, None], 1.0)
    ece = 0.0
    bins = torch.linspace(0, 1, 16, device=p.device)
    for low, high in zip(bins[:-1], bins[1:]):
        inside = (confidence > low) & (confidence <= high)
        if inside.any():
            ece += float(inside.double().mean() *
                         (confidence[inside].mean() - correct[inside].mean()).abs())
    return {"accuracy": float(correct.mean()),
            "nll": float(-p.gather(1, y[:, None]).clamp_min(1e-12).log().mean()),
            "brier": float((p - onehot).square().sum(1).mean()), "ece": ece}


def predictor(data):
    operators = {}

    def evaluate(method, params):
        probability, _ = predict(
            method, data["q"], data["edges"], params, logits=data["logits"],
            train_index=data["splits"].get("train_index"),
            train_labels=data.get("train_labels"), operators=operators)
        return probability

    return evaluate


#Compare fixed depths
@torch.no_grad()
def depth_predictions(data):
    q, logits, edges = data["q"], data["logits"], data["edges"]
    operator = graph_operator(edges, len(q), dtype=q.dtype, device=q.device)
    curves = (("appnp", 0.0), ("ppr", 0.0), ("logit_sharp", 16.0),
              ("pts", 16.0), ("pts", 200.0))
    for alpha in ALPHAS:
        for depth in DEPTHS:
            yield "anchor", {"alpha": alpha, "steps": depth, "eta": 0.0}, q
        for method, eta in curves:
            source = logits if method in {"appnp", "logit_sharp"} else q
            state = source
            for depth in range(1, max(DEPTHS) + 1):
                state = alpha * source + (1 - alpha) * torch.sparse.mm(operator, state)
                if method == "pts":
                    state = sharpen(state, eta)
                elif method == "logit_sharp":
                    state = state + eta * state.softmax(1)
                if depth in DEPTHS:
                    p = state.softmax(1) if method in {"appnp", "logit_sharp"} else normalize(state)
                    yield method, {"alpha": alpha, "steps": depth, "eta": eta}, p


def depth_rows(data):
    rows = []
    for method, params, p in depth_predictions(data):
        rows.append({"method": method, "alpha": params["alpha"], "K": params["steps"],
                     "eta": params["eta"],
                     "accuracy": accuracy(p, data["y"], data["splits"]["test_index"]),
                     "val_accuracy": accuracy(p, data["y"], data["splits"]["val_index"])})
    return rows


def energy_rows(data):
    q = data["q"]
    operator = graph_operator(data["edges"], len(q), dtype=q.dtype, device=q.device)
    row_sum = torch.sparse.mm(operator, torch.ones_like(q[:, :1])).squeeze(1)
    mass = float(row_sum.sum())
    rows = []
    for method, eta in (("ppr", 0.0), ("pts", 16.0)):
        state = q

        def record(step, phase):
            p = normalize(state)
            agreement = float((p * torch.sparse.mm(operator, p)).sum())
            weighted_square = float((row_sum[:, None] * p.square()).sum())
            dirichlet = 0.5 * (weighted_square - agreement)
            gini = 0.5 * (mass - weighted_square)
            rows.append({"method": method, "K": step, "phase": phase, "alpha": 0.1,
                         "eta": eta, "dirichlet": dirichlet, "gini": gini,
                         "potts": dirichlet + gini, "kernel_mass": mass,
                         "val_accuracy": accuracy(p, data["y"], data["splits"]["val_index"])})

        record(0, "initial")
        for step in range(1, 21):
            state = 0.1 * q + 0.9 * torch.sparse.mm(operator, state)
            record(step, "transport")
            state = sharpen(state, eta)
            record(step, "reaction")
    return rows


#Fit temperature on validation predictions, for ablation
@torch.no_grad()
def fit_temperature(probability, labels, validation):
    logp = probability[validation].clamp_min(1e-12).log()
    target = labels[validation]

    def nll(temperature):
        scaled = (logp / temperature).softmax(1)
        return float(-scaled.gather(1, target[:, None]).clamp_min(1e-12).log().mean())

    best = min(torch.logspace(-1.3, 1.6, 60).tolist(), key=nll)
    low, high = best / 1.6, best * 1.6
    for _ in range(3):
        best = min(torch.linspace(low, high, 25).tolist(), key=nll)
        width = (high - low) / 8
        low, high = max(best - width, 1e-3), best + width
    return float(best)


def calibration_rows(data, settings):
    evaluate = predictor(data)
    rows = []
    for method in ("anchor", "appnp", "pts"):
        p = evaluate(method, settings["methods"][method])
        temperature = fit_temperature(p, data["y"], data["splits"]["val_index"])
        calibrated = (p.clamp_min(1e-12).log() / temperature).softmax(1)
        raw = metrics(p, data["y"], data["splits"]["test_index"])
        scaled = metrics(calibrated, data["y"], data["splits"]["test_index"])
        rows.append({"method": method, "temperature": temperature,
                     **{f"raw_{name}": value for name, value in raw.items()},
                     **{f"cal_{name}": value for name, value in scaled.items()}})
    return rows


#Fit quintiles for local performance, per degree
def quintiles(values): 
    rank = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return np.minimum(rank * 5 // len(values), 4), ["Q1", "Q2", "Q3", "Q4", "Q5"]


#Group nodes by degree, homophily and confidence.
def per_node_rows(data, settings):
    n = len(data["q"])
    edges = graph_operator(data["edges"], n, self_loops=False, device="cpu").indices()
    source, target = edges
    labels = data["y"].detach().cpu()
    degree = torch.bincount(source, minlength=n)
    same = torch.zeros(n, dtype=torch.float64)
    same.index_add_(0, source, labels[source].eq(labels[target]).double())
    local_h = (same / degree.clamp_min(1)).float().numpy()
    test = data["splits"]["test_index"].detach().cpu().numpy()
    degree_test = degree.numpy()[test]
    degree_bins, degree_labels = quintiles(degree_test)
    boundaries = [-1e-9, .2, .4, .6, .8, 1.0 + 1e-9]
    hom_bins = np.where(degree_test == 0, 5,
                        np.clip(np.searchsorted(boundaries, local_h[test], side="right") - 1, 0, 4))
    hom_labels = ["0-.2", ".2-.4", ".4-.6", ".6-.8", ".8-1", "isolated"]
    confidence = data["q"].max(1).values.detach().cpu().numpy().astype(np.float16)[test]
    confidence_bins, confidence_labels = quintiles(confidence)
    strata = (("local_homophily", hom_bins, hom_labels),
              ("degree_quintile", degree_bins, degree_labels),
              ("anchor_confidence_quintile", confidence_bins, confidence_labels))
    evaluate = predictor(data)
    rows = []
    for method in ("appnp", "pts"):
        p = evaluate(method, settings["methods"][method])
        correct = p.argmax(1).detach().cpu().numpy()[test] == labels.numpy()[test]
        for variable, bins, names in strata:
            for index, name in enumerate(names):
                selected = bins == index
                if selected.any():
                    rows.append({"method": method, "variable": variable, "bucket": name,
                                 "bucket_index": index, "n": int(selected.sum()),
                                 "accuracy": float(correct[selected].mean())})
    return rows


#Time the comparisons
def timed(function, device, warmup, repeats):
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        function()
        sync()
    samples = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        function()
        sync()
        samples.append((time.perf_counter() - start) * 1000)
    return median(samples)


def timing_hardware(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine()


@torch.no_grad()
def timing_rows(data, forward, timing_params):
    """Time complete calls; per-step values are fixed-100-call time divided by 100."""
    device = data["q"].device
    common = {"device": str(device), "hardware": timing_hardware(device),
              "threads": torch.get_num_threads(), "dtype": str(data["q"].dtype)}
    forward_ms = timed(forward, device, warmup=1, repeats=3)
    rows = [{**common, "method": "backbone", "mode": "forward", "ms": forward_ms,
             "ms_per_step": None, "K": 0, "parameters": {},
             "training_seconds": getattr(forward, "training_seconds", None)}]
    if data["metadata"]["backbone"] != "mlp":
        return rows
    required = ("appnp", "ppr", "pts", "cs", "cs_pts")
    missing = set(required) - set(timing_params)
    if missing:
        raise ValueError("Timing needs median sigma=2 selections for: " + ", ".join(sorted(missing)))
    evaluate = predictor(data)
    for method in required:
        params = timing_params[method]
        elapsed = timed(lambda: evaluate(method, params), device, warmup=2, repeats=5)
        depth = params.get("steps", params.get("steps_correct", 0) + params.get("steps_smooth", 0))
        rows.append({**common, "method": method, "mode": "selected", "ms": elapsed,
                     "ms_per_step": None, "K": depth, "parameters": params})
    for method in ("appnp", "ppr", "pts"):
        params = {"alpha": 0.1, "steps": 100}
        if method == "pts":
            params["eta"] = 16.0
        elapsed = timed(lambda: evaluate(method, params), device, warmup=1, repeats=3)
        rows.append({**common, "method": method, "mode": "fixed_100", "ms": elapsed,
                     "ms_per_step": elapsed / 100, "K": 100, "parameters": params})
    return rows


@torch.no_grad()
def diagnostic_rows(stage, data, settings, *, forward=None, timing_params=None):
    if stage == "depth":
        return depth_rows(data)
    if stage == "energy":
        return energy_rows(data)
    if stage == "calibration":
        return calibration_rows(data, settings)
    if stage == "per_node":
        return per_node_rows(data, settings)
    if stage == "timing":
        return timing_rows(data, forward, timing_params)
