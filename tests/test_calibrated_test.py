"""Synthetic tests of the calibrated null-model test — CPU, no downloads, instant.

Encodes the paper's core claims as assertions on synthetic statistics:
  - the naive test over-fires on a null that carries spurious above-chance signal;
  - the calibrated null-model test controls the false-positive rate to ~alpha;
  - it retains power on a genuine (trained) statistic well above the null;
  - the K-seed curve: K=1 is useless, FPR converges toward alpha as K grows.
"""

import numpy as np

from deadsalmon import calibrated_test as ct


def _null_pool(mean=0.55, sd=0.02, size=40, seed=0):
    """A pool of null-model held-out accuracies: above chance (0.50) by spurious
    structure, like a random-init probe on a real dataset."""
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(mean, sd, size), 0.0, 1.0)


def test_naive_overfires_calibrated_controls():
    null_pool = _null_pool(mean=0.58, sd=0.02, size=40)
    n = 500
    naive = ct.false_positive_rate(null_pool, n, test="naive")
    calibrated = ct.false_positive_rate(null_pool, n, test="calibrated")
    # naive test flags most null findings as significant (0.58 >> chance at n=500)
    assert naive > 0.8
    # calibrated controls near alpha
    assert calibrated <= 0.15


def test_power_retained_on_genuine_signal():
    null_pool = _null_pool(mean=0.58, sd=0.02, size=40)
    trained = 0.85  # a genuine finding, well above the null band
    assert ct.power(trained, null_pool, 500, test="calibrated") is True
    assert ct.z_score(trained, null_pool) > 3


def test_calibrated_does_not_flag_the_null_itself():
    null_pool = _null_pool(mean=0.58, sd=0.02, size=40)
    # a value at the null mean must not be called significant
    assert ct.calibrated_significant(null_pool.mean(), null_pool) is False


def test_k_seed_curve_converges():
    null_pool = _null_pool(mean=0.58, sd=0.02, size=60)
    curve = ct.k_seed_curve(null_pool, k_values=(1, 2, 5, 12, 32),
                            rng=np.random.default_rng(0), n_repeats=500)
    # K=1 is useless (near 0.5), and FPR drops toward alpha as K grows
    assert curve[1] > 0.3
    assert curve[32] < curve[1]
    assert curve[32] < 0.15


def test_naive_binomial_threshold():
    # at n=500, chance-level accuracy must not be naive-significant
    assert ct.naive_significant(0.50, 500) is False
    # clearly-above-chance must be
    assert ct.naive_significant(0.60, 500) is True
