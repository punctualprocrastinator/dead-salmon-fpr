"""Feature extraction: mean-pooled hidden states from trained or null models.

Needs torch + transformers. Kept separate so the stats modules
(calibrated_test, methods) run with no deep-learning dependency.
"""

from __future__ import annotations

import numpy as np


def evenly_spaced_layers(n_layers: int, k: int = 4):
    return sorted(set(int(round(x)) for x in np.linspace(1, n_layers, k)))


def mean_pooled_features(model, tokenizer, sentences, layer: int,
                         device: str = "cuda", batch_size: int = 32,
                         max_length: int = 128):
    """Return an (N, d) array of mean-pooled residual-stream activations at
    `layer` for `sentences`. Right-pads; mean-pools over real (non-pad) tokens."""
    import torch

    feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length).to(device)
            out = model(**enc, output_hidden_states=True)
            h = out.hidden_states[layer]  # (B, T, d)
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)  # (B, T, 1)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            feats.append(pooled.float().cpu().numpy())
    return np.concatenate(feats, 0)


def load_trained(model_name: str, dtype="bfloat16", device: str = "cuda"):
    """Load a pretrained model + tokenizer. transformers 5.x uses `dtype=`."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dt = getattr(torch, dtype)
    model = AutoModel.from_pretrained(model_name, dtype=dt).to(device)
    return model, tok


def make_null_skeleton(model_name: str, dtype="bfloat16", device: str = "cuda"):
    """A random-weight model of the same architecture (no pretrained weights).
    Build once; re-randomize per seed with nulls.reinit_in_place."""
    import torch
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(model_name)
    dt = getattr(torch, dtype)
    return AutoModel.from_config(cfg).to(device=device, dtype=dt)
