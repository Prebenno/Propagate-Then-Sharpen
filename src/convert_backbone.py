"""Reuse full-package backbone weights in the minimal Backbone implementation."""

import re


def convert_state_dict(state_dict, kind, dataset):
    """Rename parameters without changing tensors, including products' sparse GNNs."""
    if kind not in {"mlp", "gcn", "sage"}:
        raise ValueError(f"Unsupported backbone: {kind}")
    converted = {}
    for name, tensor in state_dict.items():
        target = name
        if kind != "mlp" and dataset == "ogbn-products":
            target = re.sub(r"^lins\.(\d+)\.", r"layers.\1.", name)
        elif kind == "gcn":
            target = re.sub(r"^convs\.(\d+)\.lin\.weight$", r"layers.\1.weight", name)
            target = re.sub(r"^convs\.(\d+)\.bias$", r"biases.\1", target)
        elif kind == "sage":
            target = re.sub(r"^convs\.(\d+)\.lin_l\.(weight|bias)$", r"layers.\1.\2", name)
            target = re.sub(r"^convs\.(\d+)\.lin_r\.weight$", r"roots.\1.weight", target)
        if target in converted:
            raise ValueError(f"Duplicate converted parameter: {target}")
        converted[target] = tensor
    return converted
