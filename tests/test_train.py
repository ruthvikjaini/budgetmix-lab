import numpy as np

from budgetmix.env import BUDGET_GRID, TMAX, make_layouts
from budgetmix.policy import init_params
from budgetmix.train import (
    Trajectories,
    group_advantages,
    grpo_grad,
    rollouts,
    sample_budget,
)


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_budget_grid_is_ten_percent_steps():
    assert BUDGET_GRID == (6, 12, 18, 24, 30, 36, 42, 48, 54, 60)
    assert BUDGET_GRID[0] == int(np.ceil(0.1 * TMAX))
    assert BUDGET_GRID[-1] == TMAX


def test_sample_budget_uniform_over_grid():
    rng = _rng(0)
    draws = [sample_budget(rng) for _ in range(2000)]
    assert set(draws) <= set(BUDGET_GRID)
    assert set(draws) == set(BUDGET_GRID)  # every budget reachable
    counts = np.bincount([BUDGET_GRID.index(d) for d in draws], minlength=10)
    assert counts.min() > 100  # roughly uniform, no budget starved


def test_group_advantages_zero_mean_and_zero_when_tied():
    r = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    adv = group_advantages(r, group=4)
    assert abs(adv[:4].mean()) < 1e-12
    assert abs(adv[4:].mean()) < 1e-12  # second group is all 1s -> std 0 -> 0
    assert np.all(adv[4:] == 0.0)
    assert adv[0] > 0 and adv[1] < 0


def _fake_traj(n=4, T=3):
    rng = _rng(0)
    obs = rng.normal(size=(n, T, 42))
    actions = rng.integers(0, 6, size=(n, T))
    logp_old = np.full((n, T), -np.log(6))
    rewards = np.array([1.0, 0.0, 1.0, 0.0])
    mask = np.ones((n, T), dtype=bool)
    states = np.zeros((n, T), dtype=int)
    return Trajectories(obs, actions, logp_old, rewards, mask, states)


def test_grpo_grad_zero_when_advantages_zero_and_no_kl():
    params = init_params(_rng(0))
    traj = _fake_traj()
    adv = np.zeros(4)
    grads, info = grpo_grad(params, traj, adv, params, beta=0.0)
    for g in grads.values():
        assert np.allclose(g, 0.0, atol=1e-12)
    assert info["clip_frac"] == 0.0


def test_grpo_grad_only_kl_when_advantages_zero():
    params = init_params(_rng(0))
    ref = init_params(_rng(1))
    traj = _fake_traj()
    adv = np.zeros(4)
    grads, info = grpo_grad(params, traj, adv, ref, beta=0.02)
    assert info["kl"] != 0.0
    assert any(np.abs(g).max() > 0 for g in grads.values())


def test_rollouts_respect_budgets():
    rng = _rng(0)
    layouts = make_layouts(2, seed=7)[:4]
    params = init_params(_rng(0))
    traj = rollouts(params, layouts, [5, 10, 5, 10], rng)
    assert traj.rewards.shape == (4,)
    assert set(np.unique(traj.rewards)) <= {0.0, 1.0}
    assert traj.mask.sum(axis=1).max() <= 10
    # Trajectories with budget 5 stop after at most 5 steps.
    assert traj.mask[0].sum() <= 5 and traj.mask[2].sum() <= 5
