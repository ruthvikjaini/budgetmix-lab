"""Demo: tiny end-to-end run of the BudgetMix reproduction.

Trains a BC base on a handful of layouts, then a short GRPO-Full and a short
GRPO-BudgetMix arm, and prints Pass@(8, T) at 25% and 100% of the budget.
Runs in about a minute on CPU.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from budgetmix.env import TMAX, make_layouts, solve_qstar
from budgetmix.evaluate import pass_at_kt
from budgetmix.train import collect_bc_data, train_bc, train_grpo


def main():
    rng = np.random.default_rng(0)
    train = make_layouts(4, seed=1000)
    ev = make_layouts(4, seed=3000)
    X, y = collect_bc_data(train, rng, n_rollouts_per_layout=2)
    base, info = train_bc(X, y, seed=0, epochs=60)
    print(f"BC agreement on expert actions: {info['bc_agreement']:.3f}")
    val = make_layouts(2, seed=2000)
    q_val = {id(l): solve_qstar(l) for l in val}
    full, _ = train_grpo(base, train, "full", seed=0, steps=20,
                         val_layouts=val, val_qstars=q_val, log_every=20)
    mix, _ = train_grpo(base, train, "budgetmix", seed=0, steps=20,
                        val_layouts=val, val_qstars=q_val, log_every=20)
    print(f"{'method':<10} {'Pass@8, T=15':>13} {'Pass@8, T=60':>13}")
    for name, params in (("base", base), ("full", full), ("budgetmix", mix)):
        p10 = pass_at_kt(params, ev, 15, k=8, rng=rng)["pass_k"]
        p40 = pass_at_kt(params, ev, 60, k=8, rng=rng)["pass_k"]
        print(f"{name:<10} {p10:>13.3f} {p40:>13.3f}")


if __name__ == "__main__":
    main()
