# Dead Salmons Have a False-Positive Rate

**A calibrated null-model test for interpretability, and a benchmark showing how badly the naive
alternative fails.**

> ⚠️ **Private / under review.** This repo accompanies a paper under double-blind review; keep it
> private (or use an anonymized mirror for the review copy) until reviews are in.

In 2009 a dead salmon in an fMRI scanner showed "brain activity" because the analysis lacked a
multiple-comparisons control. Interpretability methods have the same problem: a probe, a
concept-direction, or an SAE feature can produce a confident "the model represents X" finding on a
network that has learned nothing. This repo measures how often that happens and validates the fix.

**Headline results** (see [`docs/paper.md`](docs/paper.md)):
- Across **6 models** (124M–9B; GPT-2, BERT, Qwen2.5-0.5B/7B, Llama-3.1-8B, Gemma-2-9B), **3 method
  families** (linear probing, concept-directions, SAE feature selection), and **3 concept types**, a
  naive significance test flags **up to 100%** of findings on *untrained* models as significant.
- The false positive **worsens with model scale** and is not fixed by regularization or more data.
- It is invisible to the **shuffled-label control** practitioners run (which passes at ~2.5%).
- A **calibrated null-model test** — compare the finding against the distribution of the same
  statistic on K matched untrained models — brings the false-positive rate to near-nominal while
  retaining power, provided **K ≥ 8–12** (K = 1–2 is useless).

## The drop-in test

The core is dependency-light (numpy/scipy/sklearn), so you can add it to any pipeline that produces
a scalar statistic:

```python
from deadsalmon import calibrated_test as ct

# statistic on the trained model, and the same statistic on K matched null models
significant = ct.calibrated_significant(trained_stat, null_stats)     # vs a naive chance test
fpr        = ct.false_positive_rate(null_stats, n, test="calibrated") # measure your control
curve      = ct.k_seed_curve(null_pool)                               # "how many nulls?"
```

Reproduce the synthetic demonstration in seconds (CPU, no downloads):

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest tests/ -q      # 5 tests, encode the core claims
```

## Layout

```
src/deadsalmon/
  calibrated_test.py   the naive vs calibrated tests, FPR/power, the K-curve  (numpy/scipy only)
  methods.py           the three method statistics (probe, cav, sae_max_abs_r)
  nulls.py             null constructors (shuffled, random-direction, model reinit)
  extract.py           mean-pooled feature extraction, trained/null loaders     (torch/transformers)
  data.py              balanced binary concept datasets                          (datasets)
experiments/run_breadth.py   reproduce one model's breadth (GPU for the large models)
tests/                 synthetic self-tests (CPU, instant)
results/               archived result JSONs behind every number in the paper
figures/               result figures
docs/paper.md          the full write-up
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
