"""Post-hoc graph TV adapted from Yang et al
Solves <-log Q, U> + epsilon <U, log U> + lambda sum_{i,c} ||(S_ij (U_ic - U_jc))_j||_2.
S is the symmetrically normalized graph without self-loops.
"""

import math
import time

import torch


#Weighted edges in both directions, without duplicates or self-loops.
class GraphTVOperator:

    def __init__(
        self,
        edge_index,
        num_nodes,
        *,
        dtype=torch.float64,
        device=None,
        chunk_edges=262144,
    ):
        self.device = torch.device(device) if device is not None else edge_index.device
        self.dtype = dtype
        self.num_nodes = num_nodes
        self.chunk_edges = chunk_edges

        edges = edge_index.to(device=self.device, dtype=torch.long)
        source = torch.minimum(edges[0], edges[1])
        target = torch.maximum(edges[0], edges[1])
        keep = source != target
        edge_keys = torch.unique(source[keep] * num_nodes + target[keep], sorted=True)
        source = edge_keys.div(num_nodes, rounding_mode="floor")
        target = edge_keys % num_nodes
        self.source = torch.cat((source, target))
        self.target = torch.cat((target, source))
        self.num_edges = int(self.source.numel())

        degree = torch.bincount(self.source, minlength=num_nodes).to(dtype)
        self.weights = (degree[self.source] * degree[self.target]).rsqrt()

        weighted_degree = torch.zeros(num_nodes, dtype=torch.float64, device=self.device)
        weighted_degree.index_add_(0, self.source, self.weights.double().square())
        self.norm_squared_bound = 4.0 * float(weighted_degree.max())

    def chunks(self):
        for start in range(0, self.num_edges, self.chunk_edges):
            yield slice(start, min(start + self.chunk_edges, self.num_edges))

    #Compute weighted differences between neighbors.
    def gradient(self, probabilities):
        differences = torch.empty(
            (self.num_edges, probabilities.shape[1]),
            dtype=probabilities.dtype,
            device=self.device,
        )
        for edge_slice in self.chunks():
            source = self.source[edge_slice]
            target = self.target[edge_slice]
            differences[edge_slice] = self.weights[edge_slice, None] * (
                probabilities[source] - probabilities[target]
            )
        return differences

    #Add weighted dual values back to the nodes.
    def adjoint(self, dual, *, accumulation_dtype=None):
        dtype = accumulation_dtype or dual.dtype
        divergence = torch.zeros(
            (self.num_nodes, dual.shape[1]), dtype=dtype, device=self.device
        )
        for edge_slice in self.chunks():
            weighted_dual = dual[edge_slice].to(dtype) * self.weights[edge_slice, None].to(dtype)
            divergence.index_add_(0, self.source[edge_slice], weighted_dual)
            divergence.index_add_(0, self.target[edge_slice], -weighted_dual)
        return divergence

    #Compute an L2 norm over neighbors for each node and class.
    def group_norms(self, values):
        squared_norms = torch.zeros(
            (self.num_nodes, values.shape[1]), dtype=torch.float64, device=self.device
        )
        for edge_slice in self.chunks():
            squared_norms.index_add_(
                0, self.source[edge_slice], values[edge_slice].double().square()
            )
        return squared_norms.sqrt_()

    #Sum neighborhood norms in edge chunks.
    def total_variation(self, probabilities):
        squared_norms = torch.zeros(
            (self.num_nodes, probabilities.shape[1]),
            dtype=torch.float64,
            device=self.device,
        )
        for edge_slice in self.chunks():
            source = self.source[edge_slice]
            target = self.target[edge_slice]
            difference = probabilities[source].double() - probabilities[target].double()
            difference *= self.weights[edge_slice, None].double()
            squared_norms.index_add_(0, source, difference.square())
        return squared_norms.sqrt_().sum()

    #Project each neighborhood dual onto its L2 ball in place.
    def project(self, dual, radius):
        scales = (self.group_norms(dual) / radius).clamp_min_(1)
        for edge_slice in self.chunks():
            dual[edge_slice].div_(scales[self.source[edge_slice]].to(dual.dtype))
        return dual

    def estimated_memory_bytes(self, num_classes):
        scalar_bytes = torch.empty((), dtype=self.dtype).element_size()
        dual_bytes = 2 * self.num_edges * num_classes * scalar_bytes
        graph_bytes = self.num_edges * (16 + scalar_bytes)
        node_bytes = 10 * self.num_nodes * num_classes * 8
        scratch_bytes = 6 * min(self.num_edges, self.chunk_edges) * num_classes * 8
        return {
            "dual_buffers": dual_bytes,
            "graph": graph_bytes,
            "node_workspace_upper_estimate": node_bytes,
            "edge_workspace_upper_estimate": scratch_bytes,
            "estimated_total": dual_bytes + graph_bytes + node_bytes + scratch_bytes,
        }


#Compute fidelity and entropy in float64, using 0 * log(0) = 0 to avoid nan values.
def _unary(probabilities, log_q, epsilon):
    total = torch.zeros((), dtype=torch.float64, device=probabilities.device)
    for start in range(0, len(probabilities), 65536):
        node_slice = slice(start, start + 65536)
        current = probabilities[node_slice].double()
        anchor_log = log_q[node_slice].double()

        fidelity = torch.where(current == 0, torch.zeros_like(current), -current * anchor_log)
        entropy = epsilon * torch.special.xlogy(current, current)
        total += (fidelity + entropy).sum()
    return total


#Compute the primal-dual gap.
def _certificate(log_q, operator, dual, lam, epsilon):
    divergence = operator.adjoint(dual, accumulation_dtype=torch.float64)
    logits = (log_q - divergence) / epsilon
    dual_objective = float(-epsilon * torch.logsumexp(logits, dim=1).sum())
    probabilities = torch.softmax(logits, dim=1)
    primal_objective = float(
        _unary(probabilities, log_q, epsilon)
        + lam * operator.total_variation(probabilities)
    )

    gap = primal_objective - dual_objective
    scale = max(1.0, abs(primal_objective), abs(dual_objective))
    max_dual_norm = float(operator.group_norms(dual).max())
    feasible = max_dual_norm <= lam * (1 + 1e-12)
    certificate_values = [primal_objective, dual_objective, gap, max_dual_norm]
    if not all(math.isfinite(value) for value in certificate_values):
        raise FloatingPointError("Graph-TV produced a non-finite convergence certificate")
    if gap < -1e-9 * scale or not feasible:
        raise RuntimeError("Graph-TV produced an invalid primal-dual certificate")

    gap = max(0.0, gap)
    diagnostics = {
        "objective": primal_objective,
        "dual_objective": dual_objective,
        "primal_dual_gap": gap,
        "relative_primal_dual_gap": gap / scale,
        "dual_feasible": feasible,
        "max_dual_norm": max_dual_norm,
    }
    return probabilities, diagnostics


#Solve graph TV by accelerated projected descent.
@torch.no_grad()
def graph_tv(
    q,
    operator,
    lam,
    *,
    epsilon=1.0,
    max_iterations=10000,
    gap_tolerance=1e-5,
    gap_interval=25,
    dual_init=None,
    return_dual=False,
):
    started = time.monotonic()
    log_q = q.double().log()
    log_q -= torch.logsumexp(log_q, dim=1, keepdim=True)
    diagnostics = {
        "method": "yang_neighborhood_tv_posthoc",
        "lambda": float(lam),
        "epsilon": float(epsilon),
        "weighting": "symmetric_without_self_loops",
        "directed_edges": operator.num_edges,
        "dual_dtype": str(operator.dtype),
        "certificate_dtype": "torch.float64",
        "gap_tolerance": gap_tolerance,
        "max_iterations": max_iterations,
        "warm_start": dual_init is not None,
        "memory_estimate_bytes": operator.estimated_memory_bytes(q.shape[1]),
    }

    if lam == 0 or operator.num_edges == 0:
        probabilities = q.clone() if epsilon == 1 else torch.softmax(log_q / epsilon, dim=1)
        objective = float(_unary(probabilities.double(), log_q, epsilon))
        diagnostics.update({
            "iterations": 0,
            "converged": True,
            "status": "CONVERGED",
            "objective": objective,
            "dual_objective": objective,
            "primal_dual_gap": 0.0,
            "relative_primal_dual_gap": 0.0,
            "dual_feasible": True,
            "max_dual_norm": 0.0,
            "restarts": 0,
            "seconds": time.monotonic() - started,
        })
        if return_dual:
            return probabilities, diagnostics, None
        return probabilities, diagnostics

    step_size = 0.99 * (2 * epsilon) / operator.norm_squared_bound
    projection_radius = float(lam) * (1 - 64 * torch.finfo(operator.dtype).eps)
    if dual_init is None:
        dual = torch.zeros(
            (operator.num_edges, q.shape[1]), dtype=operator.dtype, device=q.device
        )
    else:
        dual = dual_init.to(dtype=operator.dtype, device=q.device).clone()
        operator.project(dual, projection_radius)

    candidate = torch.zeros_like(dual)
    divergence = operator.adjoint(dual, accumulation_dtype=torch.float64)
    previous_divergence = divergence.clone()
    momentum = 1.0
    extrapolation = 0.0
    restarts = 0
    objective = float(
        epsilon * torch.logsumexp((log_q - divergence) / epsilon, dim=1).sum()
    )

    #Update and project the dual variables.
    def take_projected_step(search_probabilities, extrapolation_weight):
        squared_norms = torch.zeros(
            (operator.num_nodes, q.shape[1]), dtype=torch.float64, device=q.device
        )
        for edge_slice in operator.chunks():
            previous = candidate[edge_slice].double()
            current = dual[edge_slice].double()
            source = operator.source[edge_slice]
            target = operator.target[edge_slice]
            difference = search_probabilities[source] - search_probabilities[target]
            updated = current + extrapolation_weight * (current - previous)
            updated.add_(
                difference * operator.weights[edge_slice, None].double(), alpha=step_size
            )
            candidate[edge_slice].copy_(updated)
            squared_norms.index_add_(0, source, candidate[edge_slice].double().square())

        scales = (squared_norms.sqrt_() / projection_radius).clamp_min_(1)
        for edge_slice in operator.chunks():
            candidate[edge_slice].div_(scales[operator.source[edge_slice]].to(operator.dtype))

    for iteration in range(1, max_iterations + 1):
        search_divergence = divergence + extrapolation * (divergence - previous_divergence)
        search_probabilities = torch.softmax((log_q - search_divergence) / epsilon, dim=1)
        take_projected_step(search_probabilities, extrapolation)
        candidate_divergence = operator.adjoint(candidate, accumulation_dtype=torch.float64)
        candidate_objective = float(
            epsilon * torch.logsumexp((log_q - candidate_divergence) / epsilon, dim=1).sum()
        )
        rounding_tolerance = 16 * torch.finfo(operator.dtype).eps * max(1.0, abs(objective))

        if candidate_objective > objective + rounding_tolerance:
            restarts += 1
            momentum = 1.0
            search_probabilities = torch.softmax((log_q - divergence) / epsilon, dim=1)
            take_projected_step(search_probabilities, 0.0)
            candidate_divergence = operator.adjoint(candidate, accumulation_dtype=torch.float64)
            candidate_objective = float(
                epsilon * torch.logsumexp((log_q - candidate_divergence) / epsilon, dim=1).sum()
            )
            if candidate_objective > objective + 4 * rounding_tolerance:
                raise RuntimeError("Graph-TV projected dual step increased its objective")

        dual, candidate = candidate, dual
        previous_divergence = divergence
        divergence = candidate_divergence
        objective = candidate_objective

        if iteration % gap_interval == 0 or iteration == max_iterations:
            probabilities, certificate = _certificate(log_q, operator, dual, lam, epsilon)
            if certificate["relative_primal_dual_gap"] <= gap_tolerance:
                break

        next_momentum = 0.5 * (1 + math.sqrt(1 + 4 * momentum * momentum))
        extrapolation = (momentum - 1) / next_momentum
        momentum = next_momentum

    converged = certificate["relative_primal_dual_gap"] <= gap_tolerance
    if q.device.type == "cuda":
        torch.cuda.synchronize(q.device)
    diagnostics.update(certificate)
    diagnostics.update({
        "iterations": iteration,
        "converged": converged,
        "status": "CONVERGED" if converged else "OUT_OF_ITERATIONS",
        "dual_step": step_size,
        "restarts": restarts,
        "seconds": time.monotonic() - started,
    })
    if return_dual:
        return probabilities, diagnostics, dual
    return probabilities, diagnostics
