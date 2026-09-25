"""
PtS main method here, propagate then sharpen,

Equations (2a-4) in the paper.


Nothing here is trained. pts() takes the frozen class distribution Q and the graph and alternates the propagation and sharpening, and row normalises at the end. 

eta = 0 skipts the sharpening, so PtS with eta = 0 is PPR prob. 

Sharpen() is also what C&S-PtS uses for the smoothing stage
"""

import torch


#Normalize each row, only called once on the final state.
def normalize(state):
    result = state / state.sum(1, keepdim=True) 
    return result


#Build the normalized graph matrix, Any other symmetric, non-negative affinity can be used in here instead.
def graph_operator(edge_index, num_nodes, *, dtype=torch.float64, device=None, self_loops=True):
    device = torch.device(device) if device is not None else edge_index.device
    edges = edge_index.to(device=device, dtype=torch.long)
    src, dst = edges[:, edges[0] != edges[1]]
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    keys = src * num_nodes + dst
    if self_loops:
        nodes = torch.arange(num_nodes, device=device)
        keys = torch.cat((keys, nodes * num_nodes + nodes))
    keys = torch.unique(keys, sorted=True)
    src, dst = torch.div(keys, num_nodes, rounding_mode="floor"), keys % num_nodes
    degree = torch.bincount(src, minlength=num_nodes).to(dtype)
    inv_sqrt = torch.zeros_like(degree)
    inv_sqrt[degree > 0] = degree[degree > 0].rsqrt()
    return torch.sparse_coo_tensor(
        torch.stack((src, dst)), inv_sqrt[src] * inv_sqrt[dst], (num_nodes, num_nodes),
        dtype=dtype, device=device,
    ).coalesce()


#One sharpening step, eq. (3)-(4): r_eta(p) = softmax(log p + eta p) on the normalised row, then the row mass is put back so its contribution in the next step is unchanged.
def sharpen(state, eta, epsilon=1e-12):
    if eta == 0:
        return state                      #Exact reaction off, 
    mass = state.sum(1, keepdim=True)
    p = (state / mass).clamp_min(epsilon) # Keeps log finite
    p = p / p.sum(1, keepdim=True)        #Project back to the simplex  
    return mass * torch.softmax(p.log() + eta * p, dim=1)  


#PtS, eq. (2a)-(2b). The loop keeps the unnormalised state, only the returned U^(K) is row normalized. 
#The operator uses the prebuilt S and shares it across methods.
@torch.no_grad()
def pts(q, edge_index, *, alpha, steps, eta, epsilon=1e-12, operator=None, reset_mass=False):
    if operator is None:
        operator = graph_operator(edge_index, q.shape[0], dtype=q.dtype, device=q.device)
    state = q
    for _ in range(steps):
        state = alpha * q + (1 - alpha) * torch.sparse.mm(operator, state) # (2a) restart towards frozen Q
        if reset_mass:
            state = normalize(state)
        state = sharpen(state, eta, epsilon)  #2b
    return normalize(state)
