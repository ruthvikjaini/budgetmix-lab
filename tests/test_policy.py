import numpy as np

from budgetmix.policy import (
    Adam,
    bc_loss_and_grad,
    forward,
    init_params,
    log_probs,
)


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_log_probs_normalize():
    params = init_params(_rng(0))
    X = _rng(1).normal(size=(8, 42))
    lp = log_probs(params, X)
    assert lp.shape == (8, 6)
    assert np.allclose(np.exp(lp).sum(axis=1), 1.0, atol=1e-10)


def test_bc_gradient_finite_differences():
    rng = _rng(0)
    params = init_params(rng, hidden=8)
    X = rng.normal(size=(5, 42))
    y = rng.integers(0, 6, size=5)
    loss, grads = bc_loss_and_grad(params, X, y)
    eps = 1e-6
    for key in ("W1", "b1", "W2", "b2"):
        idx = np.unravel_index(0, params[key].shape)
        orig = params[key][idx]
        params[key][idx] = orig + eps
        l_plus, _ = bc_loss_and_grad(params, X, y)
        params[key][idx] = orig - eps
        l_minus, _ = bc_loss_and_grad(params, X, y)
        params[key][idx] = orig
        num = (l_plus - l_minus) / (2 * eps)
        assert abs(num - grads[key][idx]) < 1e-5, (key, num, grads[key][idx])


def test_adam_reduces_bc_loss():
    rng = _rng(0)
    params = init_params(rng)
    X = rng.normal(size=(64, 42))
    y = (X[:, 0] > 0).astype(int)  # trivially learnable rule
    opt = Adam(params, lr=5e-3)
    l0, _ = bc_loss_and_grad(params, X, y)
    for _ in range(200):
        _, g = bc_loss_and_grad(params, X, y)
        opt.step(params, g)
    l1, _ = bc_loss_and_grad(params, X, y)
    assert l1 < l0 * 0.5
