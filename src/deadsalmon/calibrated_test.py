"""The calibrated null-model test — the paper's core contribution.

An interpretability method reports a statistic s (here: held-out accuracy of a
probe/concept-direction, or the max |Pearson r| over SAE features) that claims a
model represents a concept. The NAIVE test asks whether s beats a theoretical
chance level. The problem: untrained networks carry spurious structure, so the
naive test fires on models that have learned nothing ("dead salmons").

The CALIBRATED test instead compares s against the distribution of the same
statistic computed on K matched NULL models (untrained networks, shuffled
labels, or random directions). A finding is significant iff it exceeds that
null distribution. This controls the false-positive rate while retaining power.

The plug-in Gaussian threshold (`calibrated_significant`) under-corrects on
heavy-tailed pools, so its FPR only nears alpha by K ~ 8-12 (K=1-2 is useless).
The portable recommendation is a distribution-free EMPIRICAL-QUANTILE test
(`empirical_quantile_significant`) with K >= 20: its leave-one-out FPR is
~1/(K+1) regardless of the pool shape, so it reaches the nominal rate for
probes, concept-directions, and the extreme-value SAE statistic alike.

This module is pure numpy/scipy and has no torch/transformers dependency, so the
test is a drop-in for any pipeline that can produce a scalar statistic and a set
of null-model statistics.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

Z_CRIT_ONE_SIDED_05 = 1.645  # z for a one-sided alpha = 0.05


# --------------------------------------------------------------------------- #
# The two tests
# --------------------------------------------------------------------------- #
def naive_significant(heldout_acc: float, n: int, alpha: float = 0.05) -> bool:
    """The test practitioners implicitly run: is held-out accuracy above chance
    by a one-sided binomial test? This is what fires on dead salmons."""
    n_correct = int(round(heldout_acc * n))
    # P(X >= n_correct | p = 0.5); reject chance if that tail prob < alpha.
    p = stats.binomtest(n_correct, n, 0.5, alternative="greater").pvalue
    return bool(p < alpha)


def calibrated_significant(
    statistic: float,
    null_statistics: np.ndarray,
    z_crit: float = Z_CRIT_ONE_SIDED_05,
) -> bool:
    """The calibrated null-model test: is `statistic` above the null-model
    distribution by more than z_crit standard deviations?

    We fit a Gaussian to `null_statistics` (K matched null models) and threshold
    by z-score. The Gaussian is an approximation; with small K the empirical
    p-value floors at 1/(K+1), which is why we use the parametric z. For an
    extreme-value statistic (e.g. max-over-features) an extreme-value or
    empirical-quantile null is more appropriate — see the paper's limitations.
    """
    null_statistics = np.asarray(null_statistics, dtype=float)
    mu = null_statistics.mean()
    sd = null_statistics.std()  # population std (ddof=0): defined even at K=1 (=> 0)
    if sd == 0:
        return bool(statistic > mu)
    z = (statistic - mu) / sd
    return bool(z > z_crit)


def empirical_quantile_significant(
    statistic: float,
    null_statistics: np.ndarray,
    alpha: float = 0.05,
) -> bool:
    """Distribution-free null-model test (the paper's recommendation).

    Flag `statistic` iff it exceeds the (1 - alpha) empirical quantile of the K
    null statistics. Its leave-one-out false-positive rate is ~1/(K+1) whatever
    the pool shape, so it reaches alpha at K >= 20 (1/21 = 0.048) and, unlike the
    Gaussian plug-in, calibrates the heavy-tailed max-over-features (SAE)
    statistic too.
    """
    null_statistics = np.asarray(null_statistics, dtype=float)
    threshold = np.quantile(null_statistics, 1.0 - alpha, method="higher")
    return bool(statistic > threshold)


def z_score(statistic: float, null_statistics: np.ndarray) -> float:
    null_statistics = np.asarray(null_statistics, dtype=float)
    sd = null_statistics.std()
    if sd == 0:
        return float("inf") if statistic > null_statistics.mean() else 0.0
    return float((statistic - null_statistics.mean()) / sd)


# --------------------------------------------------------------------------- #
# Evaluating a method: false-positive rate and power
# --------------------------------------------------------------------------- #
def false_positive_rate(
    null_statistics: np.ndarray,
    n_eval: int,
    test: str = "calibrated",
    alpha: float = 0.05,
    z_crit: float = Z_CRIT_ONE_SIDED_05,
) -> float:
    """Fraction of NULL-model findings a test wrongly calls significant.

    `null_statistics` is a pool of held-out statistics from null models. For the
    naive test we ask how often each exceeds chance. For the calibrated test we
    hold each out and test it against the distribution of the rest (leave-one-out),
    which is how you would actually calibrate against a null pool.
    """
    null_statistics = np.asarray(null_statistics, dtype=float)
    K = len(null_statistics)
    if test == "naive":
        return float(np.mean([naive_significant(s, n_eval, alpha) for s in null_statistics]))
    # calibrated ("calibrated" = Gaussian z; "empirical" = distribution-free quantile),
    # leave-one-out: hold each null out and test it against the distribution of the rest.
    hits = 0
    for i in range(K):
        held = null_statistics[i]
        rest = np.delete(null_statistics, i)
        if test == "empirical":
            hits += empirical_quantile_significant(held, rest, alpha)
        else:
            hits += calibrated_significant(held, rest, z_crit)
    return hits / K


def power(
    trained_statistic: float,
    null_statistics: np.ndarray,
    n_eval: int,
    test: str = "calibrated",
    alpha: float = 0.05,
    z_crit: float = Z_CRIT_ONE_SIDED_05,
) -> bool:
    """Does the test flag a genuine trained-model finding?"""
    if test == "naive":
        return naive_significant(trained_statistic, n_eval, alpha)
    if test == "empirical":
        return empirical_quantile_significant(trained_statistic, null_statistics, alpha)
    return calibrated_significant(trained_statistic, null_statistics, z_crit)


# --------------------------------------------------------------------------- #
# The "how many nulls" curve (K recommendation)
# --------------------------------------------------------------------------- #
def k_seed_curve(
    null_pool: np.ndarray,
    k_values=(1, 2, 3, 5, 8, 12, 20, 32),
    test: str = "calibrated",
    z_crit: float = Z_CRIT_ONE_SIDED_05,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    n_repeats: int = 200,
) -> dict:
    """FPR as a function of K (number of null models used to build the null
    distribution). Demonstrates that K=1-2 is useless. With `test="calibrated"`
    (Gaussian) the FPR only nears alpha by K ~ 8-12; with `test="empirical"`
    (distribution-free quantile, the recommendation) it tracks ~1/(K+1) and
    reaches alpha at K >= 20. `null_pool` is a large pool of null-model
    statistics; for each K we repeatedly split into K "calibration" nulls and
    test a held-out null.
    """
    rng = np.random.default_rng() if rng is None else rng
    null_pool = np.asarray(null_pool, dtype=float)
    out = {}
    for K in k_values:
        if K + 1 > len(null_pool):
            continue
        hits = 0
        for _ in range(n_repeats):
            idx = rng.permutation(len(null_pool))
            cal = null_pool[idx[:K]]
            test_val = null_pool[idx[K]]
            if test == "empirical":
                hits += empirical_quantile_significant(test_val, cal, alpha)
            else:
                hits += calibrated_significant(test_val, cal, z_crit)
        out[K] = hits / n_repeats
    return out
