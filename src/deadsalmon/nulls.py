"""Null models — the "adversary moves" a genuine finding must survive.

A control is a move by an adversary who reproduces the same measurement WITHOUT
the claimed mechanism. We use three null types:

  random_init   : the same architecture with random weights (no training at all)
  shuffled      : real model, but labels permuted (no genuine label structure)
  random_dir    : project features onto a random direction (no learned direction)

For random_init on multi-billion-parameter models, re-instantiating from config
per seed is expensive; instead build one bf16 skeleton once and re-randomize it
in place per seed (see reinit_in_place), which requires clearing transformers'
`_is_hf_initialized` flag or init_weights() is a no-op on repeat calls.
"""

from __future__ import annotations

import numpy as np


def shuffled_labels(y, seed: int):
    rng = np.random.default_rng(seed)
    return rng.permutation(np.asarray(y))


def random_direction_projection(X, seed: int):
    """Project features onto a random unit direction (a 1-D 'random probe')."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    v = rng.standard_normal(d)
    v /= np.linalg.norm(v)
    return X @ v


def reinit_in_place(model, seed: int):
    """Re-randomize a HuggingFace model's weights in place (for random_init nulls
    without re-downloading/re-allocating). Clears the `_is_hf_initialized` flag so
    `init_weights()` actually re-runs, then re-inits under a fixed seed.

    Requires torch; imported lazily so the stats modules stay torch-free.
    """
    import torch

    torch.manual_seed(seed)
    for module in model.modules():
        if hasattr(module, "_is_hf_initialized"):
            module._is_hf_initialized = False
    model.init_weights()
    return model
