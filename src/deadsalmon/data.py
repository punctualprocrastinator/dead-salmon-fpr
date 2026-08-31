"""Concept datasets: balanced binary tasks. Needs `datasets`.

sentiment  = SST-2 (GLUE, nyu-mll/glue parquet — legacy loader breaks on datasets 5.x)
topic      = AG News (fancyzhx/ag_news), World vs Sports
subjectivity = SUBJ (SetFit/subj)
"""

from __future__ import annotations

import numpy as np


def _balance(texts, labels, n, seed=0):
    rng = np.random.default_rng(seed)
    texts = np.asarray(texts, dtype=object)
    labels = np.asarray(labels)
    idx0 = rng.permutation(np.where(labels == 0)[0])[: n // 2]
    idx1 = rng.permutation(np.where(labels == 1)[0])[: n // 2]
    idx = rng.permutation(np.concatenate([idx0, idx1]))
    return list(texts[idx]), labels[idx].astype(int)


def load_concept(name: str, n_train: int = 1000, n_test: int = 500, seed: int = 0):
    """Return (train_texts, y_train, test_texts, y_test), balanced 50/50."""
    from datasets import load_dataset

    if name == "sentiment":
        ds = load_dataset("nyu-mll/glue", "sst2")
        tr, te = ds["train"], ds["validation"]
        Xtr, ytr = tr["sentence"], tr["label"]
        Xte, yte = te["sentence"], te["label"]
    elif name == "topic":
        ds = load_dataset("fancyzhx/ag_news")
        # World=0, Sports=1 (of 0..3): keep those two as a binary pair
        def binp(split):
            t, y = [], []
            for r in split:
                if r["label"] in (0, 1):
                    t.append(r["text"]); y.append(r["label"])
            return t, y
        Xtr, ytr = binp(ds["train"]); Xte, yte = binp(ds["test"])
    elif name == "subjectivity":
        ds = load_dataset("SetFit/subj")
        tr, te = ds["train"], ds["test"]
        Xtr, ytr = tr["text"], tr["label"]; Xte, yte = te["text"], te["label"]
    else:
        raise ValueError(f"unknown concept {name!r}")

    Xtr, ytr = _balance(Xtr, ytr, n_train, seed)
    Xte, yte = _balance(Xte, yte, n_test, seed + 1)
    return Xtr, ytr, Xte, yte
