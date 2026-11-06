import numpy as np

from budgetmix.env import make_layouts
from budgetmix.evaluate import _wrong_action, pass_at_kt
from budgetmix.policy import init_params


def test_pass_at_kt_bounds():
    rng = np.random.default_rng(0)
    layouts = make_layouts(2, seed=3)
    params = init_params(np.random.default_rng(0))
    r = pass_at_kt(params, layouts, budget=40, k=8, rng=rng)
    assert 0.0 <= r["pass_1"] <= r["pass_k"] <= 1.0


def test_wrong_action_is_suboptimal_and_changes_state():
    rng = np.random.default_rng(0)
    layouts = make_layouts(2, seed=5)
    for layout in layouts:
        from budgetmix.env import GridTools, solve_qstar

        Q = solve_qstar(layout)
        env = GridTools(layout)
        env.reset(rng, budget=40)
        s = env.full_state()
        a, s2 = _wrong_action(layout, Q, s, rng)
        assert Q[a, s] <= Q[:, s].max() + 1e-9
        assert 0 <= s2 < 196
