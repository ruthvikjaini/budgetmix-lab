"""Main grid: BC base, GRPO-Full vs GRPO-BudgetMix (3 seeds), Pass@(K,T),
step-level quality, error-state recovery, depth profile, likelihood.
Writes results/grid.json (per-seed) and results/measured.json (aggregated).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from common import RESULTS, SEEDS, build_sets, qstars_for
from budgetmix.env import TMAX
from budgetmix.evaluate import (
    depth_quality,
    error_state_quality,
    likelihood_analysis,
    pass_at_kt,
    step_quality,
)
from budgetmix.policy import load_params, save_params
from budgetmix.train import collect_bc_data, eval_dynamics, train_bc, train_grpo

BUDGETS = [15, 30, 45, 60]  # 25/50/75/100% of TMAX


def run_seed(seed, train, val, ev, q_ev, q_val, steps, out_dir):
    rng = np.random.default_rng(10_000 + seed)
    t0 = time.time()
    # --- base policy by behavior cloning on DP-optimal actions ---
    X, y = collect_bc_data(train, rng)
    base, bc_info = train_bc(X, y, seed=seed)
    save_params(base, os.path.join(out_dir, f"base_seed{seed}.npz"))
    # --- GRPO arms from the same base ---
    full, hist_full = train_grpo(
        base, train, "full", seed=seed, steps=steps,
        val_layouts=val, val_qstars=q_val,
    )
    save_params(full, os.path.join(out_dir, f"full_seed{seed}.npz"))
    mix, hist_mix = train_grpo(
        base, train, "budgetmix", seed=seed, steps=steps,
        val_layouts=val, val_qstars=q_val,
    )
    save_params(mix, os.path.join(out_dir, f"budgetmix_seed{seed}.npz"))

    res = {"seed": seed, "bc": bc_info, "dynamics": {"full": hist_full, "budgetmix": hist_mix}}
    # --- Pass@(K,T) grid ---
    for name, params in (("base", base), ("full", full), ("budgetmix", mix)):
        for T in BUDGETS:
            res[f"pass_{name}_T{T}"] = pass_at_kt(params, ev, T, k=8, rng=rng)
    # --- step-level quality (paper sec 3.3) ---
    res["step_quality_full"] = step_quality(base, full, ev, q_ev, rng)
    res["step_quality_mix"] = step_quality(base, mix, ev, q_ev, rng)
    res["error_full"] = error_state_quality(base, full, ev, q_ev, rng)
    res["error_mix"] = error_state_quality(base, mix, ev, q_ev, rng)
    res["depth_full"] = depth_quality(base, full, ev, q_ev, rng)
    res["depth_mix"] = depth_quality(base, mix, ev, q_ev, rng)
    res["likelihood_full"] = likelihood_analysis(base, full, ev, rng)
    res["likelihood_mix"] = likelihood_analysis(base, mix, ev, rng)
    res["wall_seconds"] = float(time.time() - t0)
    return res


def aggregate(per_seed):
    """Mean across seeds for every numeric leaf."""
    agg = {}
    keys = per_seed[0].keys()
    for k in keys:
        vals = [s[k] for s in per_seed]
        if all(isinstance(v, dict) for v in vals):
            agg[k] = aggregate(vals)
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            agg[k] = float(np.mean(vals))
        elif all(isinstance(v, list) for v in vals):
            agg[k] = vals[0]
    return agg


def pass_table(per_seed):
    """Per-seed Pass@(8, T) table, kept explicit for the README cross-foot."""
    table = {}
    for name in ("base", "full", "budgetmix"):
        for T in BUDGETS:
            table[f"{name}_T{T}"] = [
                s[f"pass_{name}_T{T}"]["pass_k"] for s in per_seed
            ]
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--out", default=RESULTS)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "checkpoints"), exist_ok=True)
    train, val, ev = build_sets()
    q_val = qstars_for(val)
    q_ev = qstars_for(ev)
    per_seed = []
    for seed in SEEDS[: args.seeds]:
        res = run_seed(seed, train, val, ev, q_ev, q_val, args.steps,
                       os.path.join(args.out, "checkpoints"))
        per_seed.append(res)
        with open(os.path.join(args.out, "grid.json"), "w") as f:
            json.dump({"per_seed": per_seed}, f, indent=2)
        print(f"seed {seed} done in {res['wall_seconds']:.0f}s", flush=True)
    agg = aggregate(per_seed)
    agg["pass_table_per_seed"] = pass_table(per_seed)
    with open(os.path.join(args.out, "measured.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps(agg["pass_table_per_seed"], indent=2))


if __name__ == "__main__":
    main()
