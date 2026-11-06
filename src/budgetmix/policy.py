"""Small MLP agent policy (NumPy, manual backprop) and Adam optimizer.

Input: 42-dim partial observation. 42 -> 64 (tanh) -> 6 action logits.
No value head: GRPO uses group-relative advantages, so no critic is needed.
"""

from __future__ import annotations

import numpy as np

from .env import N_ACTIONS

OBS_DIM = 42
HIDDEN = 64


def init_params(rng: np.random.Generator, hidden: int = HIDDEN):
    scale1 = np.sqrt(2.0 / OBS_DIM)
    scale2 = np.sqrt(2.0 / hidden)
    return {
        "W1": rng.normal(0, scale1, (OBS_DIM, hidden)),
        "b1": np.zeros(hidden),
        "W2": rng.normal(0, scale2, (hidden, N_ACTIONS)),
        "b2": np.zeros(N_ACTIONS),
    }



def forward(params, X: np.ndarray):
    """X: (B, 42). Returns (logits, cache)."""
    H = np.tanh(X @ params["W1"] + params["b1"])
    logits = H @ params["W2"] + params["b2"]
    return logits, (X, H)


def log_probs(params, X: np.ndarray):
    logits, _ = forward(params, X)
    m = logits.max(axis=1, keepdims=True)
    logsum = m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
    return logits - logsum


def sample_action(params, obs: np.ndarray, rng: np.random.Generator) -> int:
    lp = log_probs(params, obs[None, :])[0]
    p = np.exp(lp)
    return int(rng.choice(N_ACTIONS, p=p / p.sum()))


def bc_loss_and_grad(params, X: np.ndarray, y: np.ndarray):
    """Cross-entropy loss and gradient for behavior cloning."""
    B = X.shape[0]
    logits, (Xc, H) = forward(params, X)
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    p = e / e.sum(axis=1, keepdims=True)
    loss = -np.log(p[np.arange(B), y] + 1e-12).mean()
    dlogits = p.copy()
    dlogits[np.arange(B), y] -= 1.0
    dlogits /= B
    gW2 = H.T @ dlogits
    gb2 = dlogits.sum(axis=0)
    dH = dlogits @ params["W2"].T
    dZ = dH * (1.0 - H ** 2)
    gW1 = Xc.T @ dZ
    gb1 = dZ.sum(axis=0)
    return loss, {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}


class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.beta1 ** self.t)
            vh = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def save_params(params, path: str) -> None:
    np.savez(path, **params)


def load_params(path: str):
    z = np.load(path)
    return {k: z[k] for k in z.files}
