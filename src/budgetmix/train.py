"""Behavior cloning (base policy) and GRPO training (full-budget vs BudgetMix).

GRPO follows the paper's setup: trajectory-level reward only (1 on goal, else
0), group-normalized advantages over G rollouts of the same task, PPO-style
clipped surrogate, small KL penalty to the base checkpoint. BudgetMix
(arXiv:2611.06483 Eq. 2) samples one training budget per task group uniformly
from the 10% grid {4, 8, ..., 40}; the budget is visible to the policy as two
observation features, exactly as the paper puts total/remaining budget in the
prompt. Trajectories that exceed their budget score 0.
"""

from __future__ import annotations

import copy

import numpy as np

from .env import (
    BUDGET_GRID,
    N_ACTIONS,
    TMAX,
    GridTools,
    expert_action,
    solve_qstar,
)
from .policy import Adam, bc_loss_and_grad, forward, log_probs


# -- batched rollouts -------------------------------------------------------

class Trajectories:
    def __init__(self, obs, actions, logp_old, rewards, mask, states):
        self.obs = obs            # (N, T, 42) float
        self.actions = actions    # (N, T) int
        self.logp_old = logp_old  # (N, T) float
        self.rewards = rewards    # (N,) float, terminal reward
        self.mask = mask          # (N, T) bool, True for real steps
        self.states = states      # (N, T) int, full-MDP state index (-1 padded)


def rollouts(params, layouts, budgets, rng, blind_budget=False):
    """Sample one trajectory per (layout, budget). Batched policy forward per
    turn. Returns a Trajectories object."""
    n = len(layouts)
    max_t = max(budgets)
    envs = [GridTools(l) for l in layouts]
    for env, b in zip(envs, budgets):
        env.reset(rng, budget=int(b))
    obs = np.zeros((n, max_t, 42))
    actions = np.zeros((n, max_t), dtype=int)
    logp_old = np.zeros((n, max_t))
    states = np.full((n, max_t), -1, dtype=int)
    mask = np.zeros((n, max_t), dtype=bool)
    rewards = np.zeros(n)
    done = np.zeros(n, dtype=bool)
    for t in range(max_t):
        active = ~done
        if not active.any():
            break
        idx = np.flatnonzero(active)
        batch_obs = np.stack(
            [envs[i].observe(budgets[i], blind_budget=blind_budget) for i in idx]
        )
        lp = log_probs(params, batch_obs)
        p = np.exp(lp)
        p = p / p.sum(axis=1, keepdims=True)
        acts = np.array([rng.choice(N_ACTIONS, p=p[j]) for j in range(len(idx))])
        for j, i in enumerate(idx):
            obs[i, t] = batch_obs[j]
            actions[i, t] = acts[j]
            logp_old[i, t] = lp[j, acts[j]]
            states[i, t] = envs[i].full_state()
            mask[i, t] = True
            r, d = envs[i].step(int(acts[j]), rng)
            if d:
                done[i] = True
                rewards[i] = r
    return Trajectories(obs, actions, logp_old, rewards, mask, states)


def group_advantages(rewards, group):
    """Group-normalized advantage, broadcast per trajectory. Groups of
    identical reward get advantage 0 (no gradient), as in GRPO."""
    n = rewards.shape[0]
    adv = np.zeros(n)
    for g in range(0, n, group):
        r = rewards[g : g + group]
        std = r.std()
        if std > 1e-8:
            adv[g : g + group] = (r - r.mean()) / std
    return adv


def _grad_from_dlogits(params, cache, dlogits):
    X, H = cache
    B = X.shape[0]
    gW2 = H.T @ dlogits
    gb2 = dlogits.sum(axis=0)
    dH = dlogits @ params["W2"].T
    dZ = dH * (1.0 - H ** 2)
    gW1 = X.T @ dZ
    gb1 = dZ.sum(axis=0)
    return {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}


def grpo_grad(params, traj: Trajectories, adv, ref_params, beta=0.02, clip=0.2):
    """Gradient of the clipped GRPO surrogate + KL penalty to ref.
    Returns (grads, info dict with mean ratio / clip fraction / KL)."""
    N, T, _ = traj.obs.shape
    X = traj.obs[traj.mask]              # (M, 42)
    a = traj.actions[traj.mask]          # (M,)
    lp_old = traj.logp_old[traj.mask]
    adv_step = np.repeat(adv, traj.mask.sum(axis=1))  # (M,)
    logits, cache = forward(params, X)
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    p = e / e.sum(axis=1, keepdims=True)
    lp = np.log(p[np.arange(X.shape[0]), a] + 1e-12)
    ratio = np.exp(lp - lp_old)
    clipped = np.clip(ratio, 1 - clip, 1 + clip)
    unclipped_term = ratio * adv_step
    clipped_term = clipped * adv_step
    use_unclipped = unclipped_term <= clipped_term
    # d(loss)/d(logp of taken action), loss = -mean(min(...)) + beta*mean(lp_ref - lp)
    lp_ref = log_probs(ref_params, X)[np.arange(X.shape[0]), a]
    kl = float(np.mean(lp_ref - lp))
    Z = X.shape[0]
    coef = np.where(use_unclipped, ratio, 0.0) * adv_step
    dlogp = -(coef + beta) / Z
    dlogits = np.zeros_like(logits)
    dlogits[np.arange(X.shape[0]), a] = dlogp
    # dlogp -> dlogits: subtract p * dlogp (softmax Jacobian), per row.
    dlogits -= p * dlogp[:, None]
    grads = _grad_from_dlogits(params, cache, dlogits)
    info = {
        "mean_ratio": float(ratio.mean()),
        "clip_frac": float((~use_unclipped).mean()),
        "kl": kl,
    }
    return grads, info


# -- behavior cloning --------------------------------------------------------

def collect_bc_data(layouts, rng, n_rollouts_per_layout=4, eps=0.15):
    """Roll out the DP-greedy expert (eps-random) and record (obs, expert
    action) pairs. Returns X (M, 42), y (M,)."""
    Xs, ys = [], []
    for layout in layouts:
        Q = solve_qstar(layout)
        for _ in range(n_rollouts_per_layout):
            env = GridTools(layout)
            env.reset(rng, budget=TMAX)
            done = False
            while not done:
                obs = env.observe(TMAX)
                s = env.full_state()
                a_exp = expert_action(Q, s, rng)
                Xs.append(obs)
                ys.append(a_exp)
                a = a_exp if rng.random() > eps else int(rng.integers(N_ACTIONS))
                _, done = env.step(a, rng)
    return np.stack(Xs), np.array(ys, dtype=int)


def train_bc(X, y, seed, epochs=300, batch=256, lr=3e-3):
    from .policy import init_params

    rng = np.random.default_rng(seed)
    params = init_params(rng)
    opt = Adam(params, lr=lr)
    n = X.shape[0]
    for ep in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            _, grads = bc_loss_and_grad(params, X[idx], y[idx])
            opt.step(params, grads)
    loss, _ = bc_loss_and_grad(params, X, y)
    # Greedy-action agreement on the training set.
    logits, _ = forward(params, X)
    acc = float((logits.argmax(axis=1) == y).mean())
    return params, {"bc_loss": float(loss), "bc_agreement": acc, "n_samples": n}


# -- GRPO --------------------------------------------------------------------

def sample_budget(rng: np.random.Generator) -> int:
    """One BudgetMix budget: uniform over the paper's 10% grid (Eq. 2)."""
    return int(rng.choice(np.array(BUDGET_GRID)))


def train_grpo(base_params, train_layouts, mode, seed, steps=200, group=8,
               batch_tasks=24, lr=3e-4, beta=0.02, clip=0.2,
               val_layouts=None, val_qstars=None, log_every=25,
               dyn_budget=15):
    assert mode in ("full", "budgetmix")
    rng = np.random.default_rng(seed)
    params = copy.deepcopy(base_params)
    ref = copy.deepcopy(base_params)
    opt = Adam(params, lr=lr)
    history = []
    for step in range(steps + 1):
        if val_layouts is not None and (step % log_every == 0 or step == steps):
            hist = eval_dynamics(params, val_layouts, val_qstars, rng,
                                 budget=dyn_budget)
            hist["step"] = step
            history.append(hist)
        if step == steps:
            break
        tasks = [train_layouts[i] for i in rng.integers(0, len(train_layouts), batch_tasks)]
        if mode == "full":
            budgets = [TMAX] * batch_tasks
        else:
            budgets = [sample_budget(rng) for _ in range(batch_tasks)]
        layouts_rep = [t for t in tasks for _ in range(group)]
        budgets_rep = [b for b in budgets for _ in range(group)]
        traj = rollouts(params, layouts_rep, budgets_rep, rng)
        adv = group_advantages(traj.rewards, group)
        grads, _ = grpo_grad(params, traj, adv, ref, beta=beta, clip=clip)
        opt.step(params, grads)
    return params, history


def eval_dynamics(params, val_layouts, val_qstars, rng, k_rollouts=4, budget=15):
    """R_multi (mean terminal reward) and R_single (mean Q* of actions taken)
    on a fixed validation set. Dynamics are measured at budget 15 (25% of
    TMAX), where the base policy has headroom: at TMAX the validation set
    saturates near 1.0 for every checkpoint and the curves carry no signal
    (pilot 2: R_multi 0.99 at step 0). Q* is per-layout exact DP."""
    layouts_rep = [l for l in val_layouts for _ in range(k_rollouts)]
    budgets = [budget] * len(layouts_rep)
    traj = rollouts(params, layouts_rep, budgets, rng)
    r_multi = float(traj.rewards.mean())
    q_vals, q_best = [], []
    for i, layout in enumerate(layouts_rep):
        Q = val_qstars[id(layout)]
        for t in range(traj.mask.shape[1]):
            if not traj.mask[i, t]:
                break
            s = int(traj.states[i, t])
            a = int(traj.actions[i, t])
            q_vals.append(Q[a, s])
            q_best.append(Q[:, s].max())
    q_vals = np.array(q_vals)
    q_best = np.array(q_best)
    return {
        "R_multi": r_multi,
        "R_single": float(q_vals.mean()),
        "R_single_regret": float((q_best - q_vals).mean()),
    }
