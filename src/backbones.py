"""Train MLP, GCN and GraphSAGE."""

import time

import torch
from torch import nn
from torch.nn import functional as F



def training_recipe(kind, dataset, epochs=None):
    recipe = dict(kind=kind, hidden=256, layers=3 if kind == "mlp" else 2,
                  dropout=0.5, learning_rate=0.01, weight_decay=0.0005,
                  max_epochs=500 if epochs is None else epochs, patience=100)
    if kind != "mlp" and dataset in {"ogbn-arxiv", "ogbn-products"}:
        recipe.update(layers=3, weight_decay=0.)
        if dataset == "ogbn-products":
            recipe["hidden"] = 128
    recipe["products_gcn"] = kind == "gcn" and dataset == "ogbn-products"
    recipe["products_sparse"] = kind != "mlp" and dataset == "ogbn-products"
    return recipe


def preprocess(x, train_index, mode="identity"):
    x = x.float()
    if mode == "standardize":
        training = x[train_index]
        return (x - training.mean(0)) / training.std(0, correction=1).clamp_min(1e-12)
    return x


def build_adjacency(edge_index, num_nodes, kind, device="cpu"):
    edges = edge_index.to(device)
    if kind == "gcn":
        nodes = torch.arange(num_nodes, device=device)
        edges = torch.cat((edges, torch.stack((nodes, nodes))), dim=1)
    row, col = edges
    degree = torch.zeros(num_nodes, device=device)
    degree.index_add_(0, row, torch.ones(row.numel(), device=device))
    if kind == "gcn":
        inverse = degree.clamp_min(1).rsqrt()
        weights = inverse[row] * inverse[col]
    else:
        weights = degree.clamp_min(1).reciprocal()[row]
    return torch.sparse_coo_tensor(edges, weights, (num_nodes, num_nodes)).coalesce()


class Backbone(nn.Module):
    #Products GCN adds bias before propagation.
    def __init__(self, in_features, out_features, recipe, adjacency=None):
        super().__init__()
        self.kind = recipe["kind"]
        self.products_gcn = recipe["products_gcn"]
        self.dropout = recipe["dropout"]
        self.register_buffer("adjacency", adjacency, persistent=False)
        dimensions = [in_features] + [recipe["hidden"]] * (recipe["layers"] - 1) + [out_features]
        pairs = list(zip(dimensions, dimensions[1:]))
        if self.kind != "mlp" and not recipe.get("products_sparse", self.products_gcn):
            from torch_geometric.nn import GCNConv, SAGEConv
            if self.kind == "gcn":
                convs = [GCNConv(a, b, cached=True) for a, b in pairs]
                self.layers = nn.ModuleList(conv.lin for conv in convs)
                self.biases = nn.ParameterList(conv.bias for conv in convs)
            else:
                convs = [SAGEConv(a, b, aggr="mean") for a, b in pairs]
                self.layers = nn.ModuleList(conv.lin_l for conv in convs)
                self.roots = nn.ModuleList(conv.lin_r for conv in convs)
        else:
            self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in pairs)
            if self.kind == "sage":
                self.roots = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in pairs)
        self.norms = nn.ModuleList(nn.BatchNorm1d(d) for d in dimensions[1:-1])

    def layer(self, i, x):
        if self.kind == "mlp":
            return self.layers[i](x)
        if self.kind == "sage":
            neighbors = torch.sparse.mm(self.adjacency, x)
            return self.layers[i](neighbors) + self.roots[i](x)
        x = torch.sparse.mm(self.adjacency, self.layers[i](x))
        return x if self.products_gcn else x + self.biases[i]

    def forward(self, x):
        for i, norm in enumerate(self.norms):
            x = F.dropout(F.relu(norm(self.layer(i, x))), self.dropout, self.training)
        return self.layer(len(self.layers) - 1, x)


#Choose the model with the best validation accuracy.
def train(x, y, indices, kind, dataset, seed, adjacency=None, device="cpu", epochs=None, split=0):
    from src.seeds import seed_everything, training_seed

    actual_seed = training_seed(seed, split)
    seed_everything(actual_seed)
    recipe = training_recipe(kind, dataset, epochs)
    x, y = x.to(device), y.to(device)
    train_index = indices["train_index"].to(device)
    val_index = indices["val_index"].to(device)
    model = Backbone(x.shape[1], int(y.max()) + 1, recipe, adjacency).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe["learning_rate"],
                                 weight_decay=recipe["weight_decay"])
    best, state, stale, best_epoch = -1., None, 0, -1
    started = time.monotonic()
    for epoch in range(recipe["max_epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x)[train_index], y[train_index])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            accuracy = float((model(x)[val_index].argmax(1) == y[val_index]).float().mean())
        if accuracy > best:
            best, stale, best_epoch = accuracy, 0, epoch
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= recipe["patience"]:
                break
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    checkpoint = dict(state_dict=state, recipe=recipe, in_features=x.shape[1],
                      out_features=int(y.max()) + 1, seed=seed, training_seed=actual_seed, best_epoch=best_epoch,
                      best_val_accuracy=best, epochs_run=epoch + 1,
                      training_seconds=time.monotonic() - started)
    return model, checkpoint


def load_checkpoint(path, edge_index, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    recipe = saved["recipe"]
    adjacency = None if recipe["kind"] == "mlp" else build_adjacency(
        edge_index, saved["num_nodes"], recipe["kind"], device)
    model = Backbone(saved["in_features"], saved["out_features"], recipe, adjacency).to(device)
    model.load_state_dict(saved["state_dict"])
    return model.eval().requires_grad_(False)
