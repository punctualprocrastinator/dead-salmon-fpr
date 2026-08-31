"""Reproduce one model's dead-salmon breadth: naive vs calibrated FPR + power,
for probe and CAV, across the three null types, on a concept. Needs a GPU box
for the multi-billion-parameter models. Small models (gpt2, bert-base) run on CPU.

Usage:
    python experiments/run_breadth.py --model gpt2 --concept sentiment --seeds 12

Writes results/breadth_<model>_<concept>.json matching the committed result files.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "src")
from deadsalmon import calibrated_test as ct  # noqa: E402
from deadsalmon import methods, nulls  # noqa: E402
from deadsalmon.data import load_concept  # noqa: E402
from deadsalmon.extract import (  # noqa: E402
    evenly_spaced_layers, load_trained, make_null_skeleton, mean_pooled_features,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--concept", default="sentiment")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    Xtr_txt, ytr, Xte_txt, yte = load_concept(args.concept)
    ytr, yte = np.array(ytr), np.array(yte)
    n_te = len(yte)

    trained, tok = load_trained(args.model, args.dtype, args.device)
    n_layers = trained.config.num_hidden_layers
    layers = evenly_spaced_layers(n_layers, 4)

    def stats_for(model, method):
        """probe/cav held-out accuracy averaged over layers for a given model."""
        accs = []
        for L in layers:
            Ftr = mean_pooled_features(model, tok, Xtr_txt, L, args.device)
            Fte = mean_pooled_features(model, tok, Xte_txt, L, args.device)
            if method == "probe":
                accs.append(methods.probe_accuracy(Ftr, ytr, Fte, yte))
            else:
                accs.append(methods.cav_accuracy(Ftr, ytr, Fte, yte))
        return accs  # per-layer

    result = {"model": args.model, "concept": args.concept, "seeds": args.seeds,
              "layers": layers, "methods": {}}

    for method in ("probe", "cav"):
        trained_accs = stats_for(trained, method)  # per-layer trained statistic
        # null distributions per null type (per layer x seed)
        null_skel = make_null_skeleton(args.model, args.dtype, args.device)
        by_null = {}
        for null_type in ("random_init", "shuffled", "randdir"):
            null_accs = []  # flattened over layer x seed
            for seed in range(args.seeds):
                if null_type == "random_init":
                    nulls.reinit_in_place(null_skel, 1000 + seed)
                    null_accs.extend(stats_for(null_skel, method))
                elif null_type == "shuffled":
                    ys = nulls.shuffled_labels(ytr, 1000 + seed)
                    for L in layers:
                        Ftr = mean_pooled_features(trained, tok, Xtr_txt, L, args.device)
                        Fte = mean_pooled_features(trained, tok, Xte_txt, L, args.device)
                        if method == "probe":
                            null_accs.append(methods.probe_accuracy(Ftr, ys, Fte, yte))
                        else:
                            null_accs.append(methods.cav_accuracy(Ftr, ys, Fte, yte))
                else:  # randdir: a random 1-D projection as the "finding"
                    for L in layers:
                        Fte = mean_pooled_features(trained, tok, Xte_txt, L, args.device)
                        proj = nulls.random_direction_projection(Fte, 1000 + seed)
                        thr = np.median(proj)
                        pred = (proj > thr).astype(int)
                        acc = max((pred == yte).mean(), 1 - (pred == yte).mean())
                        null_accs.append(float(acc))
            null_accs = np.array(null_accs)
            trained_stat = float(np.mean(trained_accs))
            by_null[null_type] = {
                "naive_fpr": ct.false_positive_rate(null_accs, n_te, "naive"),
                "calibrated_fpr": ct.false_positive_rate(null_accs, n_te, "calibrated"),
                "power_z_max": ct.z_score(trained_stat, null_accs),
                "mean_null_acc": float(null_accs.mean()),
            }
        result["methods"][method] = {"trained_mean_acc": float(np.mean(trained_accs)),
                                     "nulls": by_null}

    out = f"results/breadth_{args.model.replace('/', '_')}_{args.concept}.json"
    json.dump(result, open(out, "w"), indent=2)
    print(json.dumps(result["methods"], indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
