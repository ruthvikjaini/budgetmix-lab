"""Extensions (ours, not in the paper), from saved checkpoints:
E1. Unseen-budget interpolation: Pass@(8, T) at T = 5/15/25/35, which neither
    method trained on (BudgetMix's grid is multiples of 4).
E2. Budget-blind evaluation: the same policies evaluated with the budget cue
    removed (total-budget input fixed to 1.0, remaining fraction vs TMAX).
Writes results/extension.json.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from common import RESULTS, SEEDS, build_sets, qstars_for
from budgetmix.evaluate import pass_at_kt
from budgetmix.policy import load_params

UNSEEN = [9, 21, 33, 51]  # 15/35/55/85% of TMAX=60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=RESULTS)
    args = ap.parse_args()
    _train, _val, ev = build_sets()
    out = {"E1_unseen_budgets": {}, "E2_budget_blind": {}}
    for seed in SEEDS:
        rng = np.random.default_rng(20_000 + seed)
        ck = os.path.join(args.out, "checkpoints")
        params = {
            name: load_params(os.path.join(ck, f"{name}_seed{seed}.npz"))
            for name in ("base", "full", "budgetmix")
        }
        e1 = {}
        for name, p in params.items():
            for T in UNSEEN:
                e1[f"{name}_T{T}"] = pass_at_kt(p, ev, T, k=8, rng=rng)
        out["E1_unseen_budgets"][f"seed{seed}"] = e1
        e2 = {}
        for name, p in params.items():
            for T in (15, 60):
                e2[f"{name}_T{T}_blind"] = pass_at_kt(
                    p, ev, T, k=8, rng=rng, blind_budget=True
                )
                e2[f"{name}_T{T}_cued"] = pass_at_kt(p, ev, T, k=8, rng=rng)
        out["E2_budget_blind"][f"seed{seed}"] = e2
    # Aggregate: mean pass_k across seeds.
    for section in list(out.keys()):
        agg = {}
        keys = next(iter(out[section].values())).keys()
        for k in keys:
            agg[k] = float(np.mean([out[section][s][k]["pass_k"] for s in out[section]]))
        out[section + "_mean_pass_k"] = agg
    with open(os.path.join(args.out, "extension.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["E1_unseen_budgets_mean_pass_k"], indent=2))
    print(json.dumps(out["E2_budget_blind_mean_pass_k"], indent=2))


if __name__ == "__main__":
    main()
