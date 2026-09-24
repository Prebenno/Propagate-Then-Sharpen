"""APPNP, PPR-Prob, LAME-Graph, C&S, and the papers ablations"""

import torch

from .graph_tv import GraphTVOperator, graph_tv
from .pts import graph_operator, normalize, pts, sharpen


def _operator(x, edge_index, operator, *, self_loops=True):
    if operator is not None:
        return operator
    return graph_operator(edge_index, x.shape[0], dtype=x.dtype, device=x.device,
                          self_loops=self_loops)


#Propagate logits, then apply softmax.
@torch.no_grad()
def appnp(logits, edge_index, *, alpha, steps, operator=None):
    operator = _operator(logits, edge_index, operator)
    state = logits
    for _ in range(steps):
        state = alpha * logits + (1 - alpha) * torch.sparse.mm(operator, state)
    return state.softmax(1)


#Propagate probabilities without sharpening.
def ppr(q, edge_index, *, alpha, steps, operator=None, reset_mass=False):
    return pts(q, edge_index, alpha=alpha, steps=steps, eta=0, operator=operator,
               reset_mass=reset_mass)


#Sharpen logits after each APPNP step.
@torch.no_grad()
def logit_sharp(logits, edge_index, *, alpha, steps, eta, operator=None):
    operator = _operator(logits, edge_index, operator)
    state = logits
    for _ in range(steps):
        state = alpha * logits + (1 - alpha) * torch.sparse.mm(operator, state)
        if eta > 0:
            state = state + eta * state.softmax(1)
    return state.softmax(1)


#Update probabilities using graph neighbors.
@torch.no_grad()
def lame(q, edge_index, *, strength, steps, operator=None):
    if strength == 0 or steps == 0:
        return q
    operator = _operator(q, edge_index, operator)
    logq, state = q.clamp_min(1e-12).log(), q
    for _ in range(steps):
        state = torch.softmax(logq + strength * torch.sparse.mm(operator, state), dim=1)
    return state


#Correct errors using training labels, then smooth.
@torch.no_grad()
def correct_and_smooth(q, edge_index, train_index, train_labels, *, alpha_correct,
                       alpha_smooth, steps_correct=50, steps_smooth=50, eta=0,
                       epsilon=1e-12, operator=None):
    operator = _operator(q, edge_index, operator, self_loops=False)
    index = train_index.to(q.device)
    if index.dtype == torch.bool:
        index = index.nonzero().flatten()
    index = index.long()
    labels = train_labels.to(device=q.device, dtype=torch.long)
    if labels.numel() == len(q):
        labels = labels[index]
    q = normalize(q)
    target = torch.zeros_like(q)
    target[index, labels] = 1
    error = torch.zeros_like(q)
    error[index] = target[index] - q[index]
    correction = error
    for _ in range(steps_correct):
        correction = (alpha_correct * torch.sparse.mm(operator, correction)
                      + (1 - alpha_correct) * error).clamp(-1, 1)
    sigma = error[index].abs().sum() / index.numel()
    scale = sigma / correction.abs().sum(1, keepdim=True)
    scale = torch.where(~torch.isfinite(scale) | (scale > 1000), torch.ones_like(scale), scale)
    initial = q + scale * correction
    initial[index] = target[index]
    state = initial
    for _ in range(steps_smooth):
        state = (alpha_smooth * torch.sparse.mm(operator, state)
                 + (1 - alpha_smooth) * initial).clamp(0, 1)
        state = sharpen(state, eta, epsilon)
    return normalize(state)


LABEL_AWARE = {"cs", "cs_pts"}
NEEDS_LOGITS = {"appnp", "logit_sharp"}


#Run chosen baseline method with its chosen fidelity term, handles both logits and probability propagaiton
def predict(name, q, edges, params, *, logits=None, train_index=None, train_labels=None,
            operators=None):
    operators = {} if operators is None else operators
    if name == "anchor":
        return q, {}
    if name == "graph_tv":
        if "tv" not in operators:
            operators["tv"] = GraphTVOperator(edges, len(q), dtype=q.dtype, device=q.device)
        settings = dict(params)
        path = settings.pop("warm_start_lambdas", [])
        if path:
            dual = None
            solver = {k: v for k, v in settings.items()
                      if k not in {"lam", "dual_init", "return_dual"}}
            for strength in path:
                _, _, dual = graph_tv(q, operators["tv"], strength, dual_init=dual,
                                      return_dual=True, **solver)
            settings["dual_init"] = dual
        return graph_tv(q, operators["tv"], **settings)
    kind = "cs" if name in LABEL_AWARE else "sym"
    if kind not in operators:
        operators[kind] = graph_operator(edges, len(q), dtype=q.dtype, device=q.device,
                                         self_loops=kind == "sym")
    settings = dict(params, operator=operators[kind])
    if name in NEEDS_LOGITS:
        if name == "appnp":
            return appnp(logits, edges, **settings), {}
        return logit_sharp(logits, edges, **settings), {}
    if name in {"ppr", "ppr_rn"}:
        return ppr(q, edges, reset_mass=name == "ppr_rn", **settings), {}
    if name in {"pts", "pts_rn"}:
        return pts(q, edges, reset_mass=name == "pts_rn", **settings), {}
    if name == "lame":
        return lame(q, edges, **settings), {"iterations": settings["steps"], "stopping": "fixed"}
    if name in LABEL_AWARE:
        return correct_and_smooth(q, edges, train_index, train_labels, **settings), {}
