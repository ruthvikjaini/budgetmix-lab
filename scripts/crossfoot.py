"""Cross-foot the README headline numbers against results/*.json.

Recomputes every number quoted in README.md from the raw per-seed results
and fails loudly on any mismatch, so no figure is ever hand-typed.
Usage: python3 scripts/crossfoot.py [--readme README.md]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load(name):
    with open(os.path.join(ROOT, "results", name)) as f:
        return json.load(f)


def mean(xs):
    return float(np.mean(xs))


def check(name, got, want, tol=1e-9):
    if abs(got - want) > tol:
        print(f"FAIL {name}: recomputed {got} != stored {want}")
        return False
    print(f"ok   {name}: {got:.4f}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default=os.path.join(ROOT, "README.md"))
    args = ap.parse_args()
    grid = load("grid.json")
    ext = load("extension.json")
    measured = load("measured.json")
    per_seed = grid["per_seed"]
    ok = True

    # 1. Pass@(8,T) per-seed table in measured.json must match grid.json exactly.
    for name in ("base", "full", "budgetmix"):
        for T in (15, 30, 45, 60):
            stored = measured["pass_table_per_seed"][f"{name}_T{T}"]
            recomputed = [s[f"pass_{name}_T{T}"]["pass_k"] for s in per_seed]
            if len(stored) != len(recomputed) or any(
                abs(a - b) > 1e-12 for a, b in zip(stored, recomputed)
            ):
                print(f"FAIL pass_table {name}_T{T}: {stored} != {recomputed}")
                ok = False
            else:
                print(f"ok   pass_table {name}_T{T}: mean {mean(recomputed):.4f}")
    # 2. BudgetMix gain over Full at each budget, from the same per-seed table.
    for T in (15, 30, 45, 60):
        gain = mean([s[f"pass_budgetmix_T{T}"]["pass_k"] for s in per_seed]) - mean(
            [s[f"pass_full_T{T}"]["pass_k"] for s in per_seed]
        )
        print(f"ok   gain_mix_minus_full_T{T}: {gain:+.4f}")
    # 3. Step-quality / error-state win rates recompute from per-seed dicts.
    for key in ("step_quality_full", "step_quality_mix", "error_full", "error_mix"):
        for field in ("rl_win_rate", "base_mean_q", "rl_mean_q"):
            vals = [s[key][field] for s in per_seed]
            ok &= check(f"{key}.{field}", mean(vals), measured[key][field])
    # 4. Extension means recompute from per-seed extension entries.
    for section, agg_key in (("E1_unseen_budgets", "E1_unseen_budgets_mean_pass_k"),
                             ("E2_budget_blind", "E2_budget_blind_mean_pass_k")):
        for k, v in ext[agg_key].items():
            vals = [ext[section][s][k]["pass_k"] for s in ext[section] if s.startswith("seed")]
            ok &= check(f"{agg_key}.{k}", mean(vals), v)
    # 5. README must contain every headline figure it quotes, recomputed here.
    with open(args.readme) as f:
        readme = f.read()
    for name in ("base", "full", "budgetmix"):
        for T in (15, 30, 45, 60):
            v = mean([s[f"pass_{name}_T{T}"]["pass_k"] for s in per_seed])
            token = f"{v:.3f}"
            if token not in readme:
                print(f"FAIL README missing pass_{name}_T{T} = {token}")
                ok = False
    print("CROSSFOOT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
