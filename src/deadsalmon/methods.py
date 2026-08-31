"""The three interpretability-method statistics we benchmark.

Each takes features X (N x d) and binary labels y and returns a scalar that the
method reports as "how well the concept is captured." The dead-salmon question is
whether this scalar is significant on a model that has learned nothing.

Pure numpy/sklearn; feature extraction (torch/transformers) is in extract.py.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def probe_accuracy(X_tr, y_tr, X_te, y_te) -> float:
    """(1) Linear probing: held-out accuracy of a logistic-regression probe."""
    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)
    clf.fit(scaler.transform(X_tr), y_tr)
    return float(clf.score(scaler.transform(X_te), y_te))


def cav_accuracy(X_tr, y_tr, X_te, y_te) -> float:
    """(2) Concept-direction / CAV: difference-of-class-means direction, FAIRLY
    oriented and thresholded on train (the fair scoring; the buggy anti-oriented
    one-sided variant is what produced our retracted 'capacity law', see paper
    Section 6). Returns held-out accuracy."""
    mu1 = X_tr[y_tr == 1].mean(0)
    mu0 = X_tr[y_tr == 0].mean(0)
    direction = mu1 - mu0
    nrm = np.linalg.norm(direction)
    if nrm == 0:
        return 0.5
    direction = direction / nrm
    proj_tr = X_tr @ direction
    proj_te = X_te @ direction
    # threshold at the midpoint of the train class-mean projections; orient so
    # class 1 is on the high side (fair, train-calibrated).
    thr = 0.5 * (proj_tr[y_tr == 1].mean() + proj_tr[y_tr == 0].mean())
    pred = (proj_te > thr).astype(int)
    acc = (pred == y_te).mean()
    return float(max(acc, 1 - acc))  # fair orientation: take the better sign


def sae_max_abs_r(feats_sae, y) -> float:
    """(3) SAE feature selection: the max |Pearson r| between any SAE feature and
    the concept label — "the feature that best captures concept c." The naive
    error is to test this winning correlation WITHOUT correcting for the search
    over many features: a multiple-comparisons dead salmon directly analogous to
    the fMRI voxels. `feats_sae` is N x M SAE activations."""
    y = np.asarray(y, dtype=float)
    y = (y - y.mean()) / (y.std() + 1e-12)
    F = feats_sae - feats_sae.mean(0, keepdims=True)
    denom = np.sqrt((F ** 2).sum(0)) * np.sqrt((y ** 2).sum()) + 1e-12
    r = (F * y[:, None]).sum(0) / denom
    return float(np.max(np.abs(r)))
