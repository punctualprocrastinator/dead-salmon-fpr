"""deadsalmon — a calibrated null-model test for interpretability.

Core (numpy/scipy/sklearn only, no torch):
    calibrated_test : the naive vs calibrated significance tests, FPR/power, K-curve
    methods         : the three method statistics (probe, cav, sae_max_abs_r)
    nulls           : null constructors (shuffled, random-direction, model reinit)

Deep-learning glue (needs torch/transformers/datasets):
    extract         : mean-pooled feature extraction, trained/null model loaders
    data            : balanced binary concept datasets
"""

from . import calibrated_test, methods, nulls

__all__ = ["calibrated_test", "methods", "nulls"]
__version__ = "0.1.0"
