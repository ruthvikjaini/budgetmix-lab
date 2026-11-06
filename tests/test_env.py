import numpy as np

from budgetmix.env import (
    DOWN,
    LEFT,
    N_ACTIONS,
    PICKUP,
    RIGHT,
    TMAX,
    UNLOCK,
    UP,
    GridTools,
    Layout,
    expert_action,
    make_layouts,
    solve_qstar,
)


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_layouts_solvable_by_dp():
    for tier in ("easy", "medium", "hard"):
        for layout in make_layouts(3, seed=42):
            if layout.tier != tier:
                continue
            Q = solve_qstar(layout)
            env = GridTools(layout)
            env.reset(_rng(), budget=TMAX)
            s0 = env.full_state()
            assert Q[:, s0].max() > 0.05, f"{tier} layout unsolvable from start"
            # Greedy rollout reaches the goal within the budget.
            rng = _rng(1)
            done = False
            while not done:
                a = expert_action(Q, env.full_state(), rng)
                _, done = env.step(a, rng)
            assert env.turn <= TMAX
            assert (env.r, env.c) == layout.goal


def test_pickup_unlock_rules():
    rng = _rng(0)
    layout = Layout(rng, "medium")
    env = GridTools(layout)
    env.reset(rng, budget=TMAX)
    # Pickup away from the key is an error and wastes the turn.
    if (env.r, env.c) != layout.key:
        env.step(PICKUP, rng)
        assert env.last_error and not env.has_key
    # Unlock without the key is an error.
    env.step(UNLOCK, rng)
    assert env.last_error and not env.door_open


def test_door_blocks_until_unlocked():
    rng = _rng(0)
    layout = Layout(rng, "medium")
    # Teleport next to the door and try to walk through it while closed.
    env = GridTools(layout)
    env.reset(rng, budget=TMAX)
    dr, dc = layout.door
    env.r, env.c = dr, dc - 1  # left of the door (door is at col 3, inside)
    assert not env.door_open
    env.step(RIGHT, rng)
    assert (env.r, env.c) == (dr, dc - 1), "closed door must block movement"


def test_slip_zero_is_deterministic():
    rng = _rng(0)
    layout = Layout(rng, "easy")
    env = GridTools(layout)
    env.reset(rng, budget=TMAX)
    r0, c0 = env.r, env.c
    env.step(UP, rng)
    # Either moved up or bumped into a wall; never moved sideways with slip 0.
    assert (env.r, env.c) in ((r0 - 1, c0), (r0, c0))


def test_observe_shape_and_budget_features():
    rng = _rng(0)
    layout = Layout(rng, "medium")
    env = GridTools(layout)
    env.reset(rng, budget=10)
    obs = env.observe(10)
    assert obs.shape == (42,)
    assert obs[40] == 1.0  # full remaining fraction at turn 0
    assert obs[41] == 10 / TMAX
    env.step(UP, rng)
    obs = env.observe(10)
    assert obs[40] == 0.9
    blind = env.observe(10, blind_budget=True)
    assert blind[41] == 1.0 and abs(blind[40] - (TMAX - 1) / TMAX) < 1e-12


def test_qstar_bellman_residual():
    from budgetmix.env import build_transitions

    rng = _rng(0)
    for tier in ("easy", "hard"):
        layout = Layout(rng, tier)
        Q = solve_qstar(layout)
        P, R = build_transitions(layout)
        Q2 = R + 0.95 * np.einsum("asn,n->as", P, Q.max(axis=0))
        assert np.max(np.abs(Q - Q2)) < 1e-6


def test_expert_prefers_key_then_door():
    rng = _rng(0)
    layout = Layout(rng, "hard")
    Q = solve_qstar(layout)
    env = GridTools(layout)
    env.reset(rng, budget=TMAX)
    # At the start (no key), the optimal action is never UNLOCK.
    assert expert_action(Q, env.full_state(), rng) != UNLOCK
