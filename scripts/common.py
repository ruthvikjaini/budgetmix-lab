"""Shared setup for the experiment scripts: fixed task sets, Q* cache."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from budgetmix.env import make_layouts, solve_qstar  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")

SEEDS = [0, 1, 2]
N_TRAIN_PER_TIER = 30   # 90 train layouts
N_VAL_PER_TIER = 8      # 24 validation layouts
N_EVAL_PER_TIER = 80    # 240 held-out layouts


def build_sets():
    train = make_layouts(N_TRAIN_PER_TIER, seed=1000)
    val = make_layouts(N_VAL_PER_TIER, seed=2000)
    ev = make_layouts(N_EVAL_PER_TIER, seed=3000)
    return train, val, ev


def qstars_for(layouts):
    return {id(l): solve_qstar(l) for l in layouts}


def tier_of(layout):
    return layout.tier
