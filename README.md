# Dead Salmons Have a False-Positive Rate

**A calibrated null-model test for interpretability, and a benchmark showing how badly the naive
alternative fails.**

> Anonymized code for double-blind review. Author, institution, and hosting information have been
> removed; please do not attempt to de-anonymize.

In 2009 a dead salmon in an fMRI scanner showed "brain activity" because the analysis lacked a
multiple-comparisons control. Interpretability methods have the same problem: a probe, a
concept-direction, or an SAE feature can produce a confident "the model represents X" finding on a
network that has learned nothing. This repo measures how often that happens and validates the fix.

## Headline results

- Across **6 models** (124M–9B; GPT-2, BERT, Qwen2.5-0.5B/7B, Llama-3.1-8B, Gemma-2-9B), **3 method
  families** (linear probing, concept-directions / CAVs, SAE feature selection — including two
  released production SAEs), and **3 concept types**, a naive significance test flags **up to 100%**
  of findings on *untrained* models as significant.
- The failure is invisible to the **shuffled-label control** practitioners run: it passes at
  **~2.5–7%** under the *same* naive test that hits ~100% on random-init nulls — passing the control
  you run gives false reassurance about the null you did not.
- A **calibrated null-model test** compares the finding against the distribution of the same
  statistic on *K* matched untrained models. The plug-in Gaussian threshold under-corrects on
  heavy-tailed pools, so the **portable recommendation is a distribution-free empirical-quantile test
  with K ≥ 20** (the Gaussian z-test with K ≈ 8–12 is a rough approximation; K = 1–2 is useless).
  Validated leave-one-out with bootstrap CIs for probes and concept-directions across five families
  up to 7B.
- Applied to a released-SAE max-over-features statistic, the calibrated test **exposes that statistic
  as carrying essentially no signal** under a fair control — the trained model does not separate from
  random — a negative result the naive test hides. The difference-of-means CAV is similarly weak.
- We also catch a dead salmon in our **own** pipeline: a scoring convention fabricated an apparent
  "capacity law" that vanished under a fair control.

## The drop-in test

The core is dependency-light (numpy/scipy/sklearn), so you can add it to any pipeline that produces
a scalar statistic and a pool of the same statistic on matched null models:

```python
from deadsalmon import calibrated_test as ct

# recommended: distribution-free empirical-quantile test (use K >= 20 null models)
significant = ct.calibrated_test.empirical_quantile_significant(trained_stat, null_stats)

# measure a test's false-positive rate on a null pool (leave-one-out)
fpr   = ct.false_positive_rate(null_stats, n, test="empirical")   # or "calibrated" / "naive"
curve = ct.k_seed_curve(null_pool, test="empirical")              # "how many nulls?"
```

Reproduce the synthetic demonstration in seconds (CPU, no downloads):

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest tests/ -q      # 5 tests, encode the core claims
```

## Layout

```
src/deadsalmon/
  calibrated_test.py   naive / Gaussian-calibrated / empirical-quantile tests, FPR, power, K-curve
  methods.py           the three method statistics (probe, cav, sae_max_abs_r)
  nulls.py             null constructors (shuffled, random-direction, model reinit)
  extract.py           mean-pooled feature extraction, trained/null loaders     (torch/transformers)
  data.py              balanced binary concept datasets                          (datasets)
experiments/run_breadth.py   reproduce one model's breadth (GPU for the large models)
tests/                 synthetic self-tests (CPU, instant)
results/               archived result JSONs behind every number in the paper
figures/               result figures
```

## Reproducing the model experiments

`experiments/run_breadth.py` regenerates a model's naive/calibrated FPR + power. Small models
(gpt2, bert-base) run on CPU; the multi-billion-parameter models need a GPU. The committed
`results/*.json` are the exact numbers the paper reports.

```bash
python experiments/run_breadth.py --model gpt2 --concept sentiment --seeds 12
```

## Note on reconstruction

The original experiments ran in a notebook; this is a clean reference implementation rebuilt from
the archived results and documented protocol. The committed `results/` JSONs are the originals; the
`src/` code reproduces them up to random-init seed variation.

## License

MIT — see [LICENSE](LICENSE).
