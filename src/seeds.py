"""Recorded paper random streams (hashing here identifies RNG streams, not files)."""

import hashlib
import random

import numpy as np
import torch

SPLIT_SEED_BASE = 20250901
TRAINING_SEED_BASE = 20260826
STREAM_TAG = "optuna_board_v2"
CLEAN_DRAW = -1
MODEL_SEEDS = (0, 1, 2)
NOISY_DRAWS = (0, 1, 2)


def stable_seed(*parts, bits=63):
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & ((1 << bits) - 1)


def split_seed(dataset, split, base=SPLIT_SEED_BASE):
    payload = f"{int(base)}|{dataset}|split{int(split)}".encode()
    return int.from_bytes(hashlib.blake2s(payload, digest_size=8).digest(), "big")


def training_seed(model_seed, split, base=TRAINING_SEED_BASE):
    return int(base) + 100_003 * int(model_seed) + int(split)


def corruption_seed(dataset, split, draw, role="test", salt=STREAM_TAG):
    return stable_seed(salt, dataset, int(split), "gaussian", role, int(draw))


def study_seed(dataset, split, model_seed, sigma, draw, method, tag=STREAM_TAG):
    """Method uses its recorded name, e.g. potts rather than the CLI alias pts."""
    return stable_seed(tag, dataset, int(split), int(model_seed), float(sigma), int(draw), method, bits=32)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
