# Dead Salmons Have a False-Positive Rate: A Calibrated Null-Model Test for Interpretability

*Full workshop draft v2 (Interpretability as a Science, NeurIPS 2026). Supersedes v1: R4's
"capacity reversal" was traced to a scoring artifact and is now recast as the paper's central
cautionary example, not a finding. All numbers from the reproducible harness (archived JSONs in
results/). Citations marked [verify] are post-cutoff arXiv IDs to confirm before submission.*

---

## Abstract

Interpretability methods are used to claim what a model represents — but a method can report a
confident-looking finding on a network that has learned nothing. In the neuroscience version of this
failure, a dead salmon in an fMRI scanner showed "brain activity" because the analysis lacked a
multiple-comparisons control. We ask whether interpretability has the same problem, and measure it.
Across six models spanning a 70x parameter range and four architecture families (GPT-2, BERT,
Qwen2.5-0.5B and -7B, Llama-3.1-8B, Gemma-2-9B), three method families (linear probing,
concept-direction analysis, and sparse-autoencoder feature selection), and three concept types
(sentiment, topic, subjectivity), a naive significance test flags up to 100% of findings on
*untrained* models as significant — and, strikingly, the false-positive rate *worsens* with model
scale. The effect
survives heavy regularization and larger samples — it is not ordinary overfitting — and, crucially,
it is invisible to the shuffled-label control researchers actually use (which passes at ~2.5%). A
calibrated null-model test — comparing a finding against the distribution of the same statistic on
K matched null models, the control sketched but never validated by Méloux et al. (2025) — reduces the
false-positive rate from up to 1.00 to at most 0.13 for probing and concept-directions and 0.17 for
the SAE max-over-features statistic (approaching nominal as the number of nulls grows), while
retaining power down to genuinely near-null signals, provided K ≥ 8–12. We also report a cautionary example from our own pipeline: a scoring
convention (concept-direction sign orientation under a one-sided test) manufactured an apparent
architecture-dependent "capacity law" that reversed under fair scoring — the exact failure mode this
paper warns against, caught in our own analysis. We release the test and argue it should be standard.

---

## 1. Introduction

In 2009, neuroscientists placed a dead Atlantic salmon in an fMRI scanner, showed it photographs of
humans in social situations, and asked it to judge their emotions. Their standard analysis found a
cluster of voxels in the salmon's brain that responded significantly to the task. The salmon was
dead; the point was methodological — without a correction for multiple comparisons, a
plausible-looking, statistically significant result can be produced from pure noise.

Interpretability runs the same risk. We probe activations, find a direction that decodes a concept,
and conclude the model *represents* it. But how do we know the finding reflects something learned,
rather than structure any network of that shape carries? The control is obvious: check whether the
method produces the same finding on a model that has learned nothing. It is rarely run, and — as we
show — when run naively, it does not work.

Méloux et al. (2025) [2512.18792] recently argued this as a position: interpretability methods produce
"dead salmon" artifacts, and explanations should be treated as statistical estimates tested against
null hypotheses. They *discuss* the artifact across probing, sparse autoencoders, and causal analyses,
but *empirically* demonstrate one case (correlation and probing on a randomly initialized BERT) and
*sketch* a randomization control in an appendix without validating it: no false-positive rate, no
power, no calibration, no test of how many nulls are needed. Two concurrent works show sparse
autoencoders in particular produce interpretable-looking features on randomly initialized transformers
[Heap et al. 2501.17727; Korznikov et al. 2602.14111]; neither frames the effect as a
multiple-comparisons / selection-bias problem or supplies a calibrated test with power. A significance
test does exist for one method — TCAV [Kim et al. 2017] tests a concept-direction against
*random-concept* directions in a *trained* model — but its null is a random concept, not an untrained
model, and it gives no false-positive benchmark or sample-size guidance.

We supply that measurement, and we make four contributions:

1. **A systematic false-positive benchmark** across six models (124M–9B, four architecture families),
   three method families, and three concept types: the naive false-positive rate on untrained models
   reaches 100%, worsens with model scale, and is robust to regularization and sample size — it is not
   ordinary overfitting.
2. **Existing controls are insufficient.** The shuffled-label control (a Hewitt–Liang control task,
   which practitioners do run) passes at ~2.5% while the *same naive test* explodes to ~100% on
   random-weight nulls — passing the control you use gives false reassurance.
3. **A validated calibrated null-model test, with a sample-size recommendation.** Comparing against
   the distribution of the statistic on K matched *untrained* models controls the false-positive rate
   to near-nominal while retaining power to near-null signals; we show K ≥ 8–12 nulls are needed
   (K = 1–2 is useless), turning "run a control" into a concrete protocol. Unlike TCAV's random-concept
   test or Adebayo et al.'s qualitative randomization checks, this is a calibrated distribution over
   untrained models with measured false-positive rate, power, and a K-recommendation — the part
   Méloux et al. leave open. (For the SAE method, our novelty is the multiple-comparisons framing and
   the calibrated test, not the observation that SAEs fire on random nets, which is concurrent work.)
4. **A cautionary example from our own analysis** (Section 6). An early version of this work reported
   an architecture-dependent "capacity law." A control revealed it was an artifact of a scoring
   convention. We keep it as a worked demonstration of the paper's thesis applied to ourselves.

We do not claim to invent randomization controls — they are the backbone of Adebayo et al. (2018)'s
saliency sanity checks and Hewitt & Liang (2019)'s probing control tasks. Our contribution is to
measure the problem systematically in modern LLM interpretability and to validate a specific,
drop-in protocol.

## 2. A control is a move against an adversary

A finding claims a mechanism: *this model represents concept c*. A control is a move by an adversary
who reproduces the same measurement *without* that mechanism. The adversary's moves are null models:
a randomly initialized network (no training), the same network with shuffled labels (no genuine
label structure), or a random direction in feature space (no learned direction). A finding is robust
when no cheap adversary move reproduces it; a method's false-positive rate is how often the adversary
wins by chance. This tells us what to measure (the rate each move reproduces a "significant" finding)
and how to test (require the finding to beat the strongest available move). Sections 3–5 make this
operational.

## 3. The calibrated null-model test

Let s be the statistic a method reports — here, held-out accuracy of a probe or of a concept-direction
classifier, treating each of several layers as a separate test location (the multiple-comparisons,
"many voxels" essence of the dead-salmon critique). The **naive test** asks whether s exceeds chance
by a one-sided per-instance significance test (binomial, alpha = 0.05) — the test implicit when a
paper reports "74% accuracy, well above the 50% baseline." The **calibrated test** builds a null
distribution: it computes s on K matched null models and flags the finding significant iff its
z-score against the null distribution exceeds 1.645. The naive test compares against a *theoretical*
chance level that assumes the only source of above-chance accuracy is genuine signal; the calibrated
test compares against the *empirical* distribution of what the method produces under the null, which
absorbs the spurious structure random features carry. We report false-positive rate and power for
both.

## 4. Experimental setup

**Concepts and data.** Three binary concept types, each balanced 1000 train / 500 test: sentiment
(SST-2), topic (AG News, World vs. Sports), subjectivity (SUBJ). **Models.** Six, spanning 124M–9B and four architecture families: GPT-2 (learned pos, MHA, LayerNorm,
causal), BERT (bidirectional, LayerNorm), and four modern RoPE/RMSNorm decoders — Qwen2.5-0.5B,
Qwen2.5-7B, Llama-3.1-8B, and Gemma-2-9B. The two Qwen models share an architecture, giving a
controlled 14x scale axis. Trained = pretrained; null = same architecture, random weights (free to
build; for the multi-billion-parameter models we re-randomize a bf16 skeleton in place per seed).
**Methods.** Three families. (1) Linear probing (logistic regression on mean-pooled hidden states,
cross-validated). (2) Concept-direction / CAV (difference-of-class-means, fairly oriented and
thresholded on train). (3) Sparse-autoencoder feature selection: a small ReLU sparse autoencoder
(dictionary size 2048, ~2.7x overcomplete) trained on the model's own layer-8 residual activations,
scored by the standard practice of selecting the single feature that best captures the concept —
the naive error being to test that winning feature's correlation without correcting for the search
over 2048 features (a multiple-comparisons problem directly analogous to the fMRI voxels). We use a
model-trained SAE rather than a released one; replicating with released SAEs is future work.
**Null battery.** Random-init weights; shuffled labels; random directions. K = 12 seeds, 95% CIs.

## 5. Results

**R1 — The dead salmon reproduces on every architecture, method family, and concept.** A linear
probe or concept-direction reads the concept from an *untrained* network above chance in every
condition (mean null accuracy 0.55–0.75 vs. 0.50); under the naive test, false-positive rates on
random-init models run **0.60–1.00** (topic, the most lexically separable concept, is worst). The
SAE method is no exception: "the feature that best captures sentiment" is significant on *every*
untrained model (naive FPR **1.00**), because selecting the top feature over an overcomplete
dictionary and then testing it ignores the search — the multiple-comparisons dead salmon in its
purest form. Notably the mechanisms differ: probe/CAV inflation is driven by lexical leakage the
random architecture passively retains, while the SAE inflation is driven by in-sample selection over
many features. Two roads to the same false positive; one calibrated test (below) controls both.

**R2 — Existing controls are insufficient.** The shuffled-label control passes at naive FPR ~0.00–0.15
(often ~0.025) — it *looks* safe — while the identical naive test on random-init and random-direction
nulls reaches ~1.00. A researcher who runs only the control they know can be badly misled.

**R3 — The calibrated test fixes it, and needs K ≥ 8–12.** The calibrated null-model test brings the
false-positive rate to **0.00–0.13** for probing and concept-directions across all concepts,
architectures, and null types (mildly anticonservative — up to ~0.125 — only at the smallest sample
sizes and for the random-direction null), while flagging every genuine trained finding (power z =
6–29). For the SAE max-over-features statistic the calibrated FPR is higher (0.08–0.17): a dozen nulls
does not fully calibrate an extreme-value statistic (Section 7). The control's quality depends on the
number of nulls: calibrated FPR falls 0.49 (K=1) → 0.10 (K=5) → 0.066 (K=12) → 0.052 (K=32),
flattening toward alpha by K ≈ 8–12 (measured on one representative cell, bert/sentiment/probe). One
null model is not a control; a dozen is.

**R4 — Robust to regularization and sample size (not overfitting).** Across L2 strengths
C ∈ {0.01…10} and train sizes N ∈ {250, 500, 1000}, the naive null-FPR stays 0.81–1.00 and the
calibrated test stays ~alpha. Heavier regularization and more data do not remove the dead salmon;
only calibration does.

**R5 — Power is retained to near-null.** Degrading a genuine signal (via label noise) from accuracy
0.84 to 0.51, the calibrated test keeps flagging the finding down to accuracy ~0.59 (z = 3.5) and
releases only at the true null (0.51). It does not discard weak-but-real findings, nor false-positive
at the null.

**R6 — The false positive worsens with scale, and the fix keeps up.** Holding architecture fixed and
scaling Qwen 14x (0.5B → 7B), passive concept leakage on the untrained model *rises* at every setting
(e.g. sentiment mean null accuracy 0.54 → 0.58, topic 0.60 → 0.69), and the naive false-positive rate
rises with it (sentiment probe 0.60 → 0.92, CAV 0.67 → 1.00; the 0.5B model's mild residual advantage
vanishes at 7B). A wider residual stream offers more random directions for a concept to correlate
with by chance, so bigger untrained networks are *more* deceptive, not less. The calibrated test
continues to control the false-positive rate to ~alpha (0.04–0.13) with full power (z up to 22) at 7B:
the disease and its cure both scale.

## 6. A dead salmon in our own pipeline (cautionary example)

An earlier version of this work reported an architecture-dependent "capacity law": high-capacity
probes were fooled on LayerNorm models while low-capacity concept-directions appeared *immune*
(false-positive rate 0.00). It was an artifact of a scoring convention. Holding everything else fixed
and varying only how the concept-direction is scored, we find: an *anti-oriented, one-sided* test
(flip the direction's sign, then ask "accuracy > 0.5") gives FPR **0.00 on all three architectures** —
manufactured "immunity," because a systematically anti-oriented direction on a lexical leaker lands
below 0.5 and is never flagged. Under *fair* train-calibrated scoring the same concept-directions are
heavily inflated (FPR **1.00** on GPT-2/BERT, **0.67** on Qwen). So the "capacity immunity" was the
scoring convention, not the model, and under fair scoring the apparent architecture ordering collapses
to a weak, method-independent leakage gradient (BERT ≥ GPT-2 > Qwen). We could not exactly reconstruct
the earlier run's specific numbers — its harness and result files were lost — so we do not claim to
reproduce them; we demonstrate the *artifact class* directly: a scoring choice that fabricates an
architecture-dependent "finding," visible only once a fair control is applied. We keep this because it
is the paper's thesis applied to us. If it can happen to a paper about dead salmons, the control is
not optional.

## 7. Discussion and limitations

**A proposed norm.** Report interpretability findings against a null-model distribution (K ≥ 8–12
matched nulls), not a theoretical chance level; reviewers should ask for it. This is the operational
counterpart to Méloux et al.'s statistical reframing — they argue explanations are estimates; we give
the estimate a validated test.

**Limitations.** (i) Three method families (probe, concept-direction, SAE feature selection), but the
SAE is a small model-trained autoencoder; replicating with released SAEs, and adding attribution and
causal methods, are future work. (ii) One task family (sentence classification) across three concept
types; token-level and generative tasks untested. (iii) Our calibrated test fits a Gaussian to the
null distribution and thresholds by z-score. This is an approximation: with K = 12 it is mildly
anticonservative (up to ~0.125) at small sample sizes and for the random-direction null, and it is a
poorer fit for the SAE max-over-features statistic, which is extreme-value rather than Gaussian
distributed — hence its higher calibrated FPR (0.08–0.17). An empirical-quantile or extreme-value test,
and more nulls, would tighten both; we use the parametric z-test for simplicity and because the
empirical p-value floors at 1/(K+1). (iv) The K ≥ 8–12 recommendation and the regularization/sample-size
robustness are each measured on one representative cell (bert/sentiment/probe); we expect but do not
prove they transfer. (v) The weak residual leakage gradient across architectures is reported but not
explained; we deliberately do not over-interpret it (the earlier attempt to do so is Section 6).

**Future directions.** Two extensions are natural. First, replicating the SAE result with *released*
production SAEs (e.g. GemmaScope) rather than our small model-trained one — concurrent work already
shows released SAEs interpret random transformers [Heap et al. 2501.17727; Korznikov et al.
2602.14111], and folding that into the calibrated test is a clean next step. Second, applying the
null-model control to newer interpretability instruments whose reliability on untrained models is
untested — for example the Jacobian Lens [Anthropic 2026], which decodes a model's "verbalizable"
concepts: does it decode spurious concept vocabulary from a randomly initialized network? Every new
interpretability tool inherits the dead-salmon risk until a null-model control says otherwise.

## 8. Conclusion

Interpretability methods fire on dead salmons, and a naive significance test certifies most of those
firings as real — robustly, across methods, architectures, concepts, regularizations, and sample
sizes, and invisibly to the control practitioners run. The fix is not new in spirit but has not been
measured or validated in modern LLM interpretability: compare against a null-model distribution of at
least a dozen matched nulls, and the false-positive rate collapses to nominal while genuine findings
survive to near-null. It costs a handful of random-init models and a few lines of code. We think it
should be run by default. And we keep our own retracted "finding" in the paper as the reason: the
control caught us, too.

## References (all IDs verified against arXiv)
- Bennett, Baird, Miller & Wolford (2010), "Neural correlates of interspecies perspective taking in
  the post-mortem Atlantic Salmon: an argument for proper multiple comparisons correction." J.
  Serendipitous and Unexpected Results 1(1):1–5. (Poster, HBM 2009.)
- Méloux, Dirupo, Portet & Peyrard (2025), "The Dead Salmons of AI Interpretability." arXiv:2512.18792.
- Adebayo et al. (2018), "Sanity Checks for Saliency Maps." arXiv:1810.03292.
- Hewitt & Liang (2019), "Designing and Interpreting Probes with Control Tasks." arXiv:1909.03368.
- Leavitt & Morcos (2020), "Towards Falsifiable Interpretability Research." arXiv:2010.12016.
- Bolukbasi et al. (2021), "An Interpretability Illusion for BERT." arXiv:2104.07143.
- Belinkov (2022), "Probing Classifiers: Promises, Shortcomings, and Advances." arXiv:2102.12452.
- Elazar et al. (2021), "Amnesic Probing." arXiv:2006.00995.
- Kim et al. (2018), "Interpretability Beyond Feature Attribution: TCAV." arXiv:1711.11279.
- Heap, Lawson, Farnik & Aitchison (2025), "Automated Interpretability Metrics Do Not Distinguish
  Trained and Random Transformers." arXiv:2501.17727.
- Korznikov, Galichin, Dontsov, Rogov, Oseledets & Tutubalina (2026), "Sanity Checks for Sparse
  Autoencoders: Do SAEs Beat Random Baselines?" arXiv:2602.14111.
- Nicolson et al. (2024), "Explaining Explainability: Recommendations for Effective Use of CAVs."
  arXiv:2404.03713.

## Figure list (all archived locally)
- fig_deadsalmon_derisk.png — F1 hook (probe reads a concept from an untrained net).
- fig_deadsalmon_concepts.png — R1/R2 (naive vs calibrated FPR across concepts × models × methods).
- fig_deadsalmon_kseed.png — R3 (calibrated FPR vs K, converging to alpha by ~8–12).
- fig_power_curve.png — R5 (power retained to near-null).
- fig_robustness.png — R4 (naive FPR persists across L2 × N; calibrated controls).
- fig_deadsalmon_sae.png — R1 SAE (untrained models "find" a significant feature 100% of the time).
- fig_deadsalmon_scale.png — R6 (naive FPR and leakage rise from Qwen-0.5B to 7B; calibrated stays ~alpha).
- fig_deadsalmon_models.png — R1/R3 across all six models (124M–9B); naive near 1.0 everywhere, calibrated hugs alpha.
- fig_cav_threshold_artifact.png — Section 6 (a scoring convention fabricating a finding).
