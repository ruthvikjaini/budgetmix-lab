"""Evaluation: Pass@(K,T), step-level action quality (exact Q*), error-state
recovery, likelihood analysis, and the two extensions (unseen budgets,
budget-blind evaluation). All measurements use held-out layouts.
"""

from __future__ import annotations

import numpy as np

from .env import N_ACTIONS, TMAX, GridTools, expert_action
from .policy import log_probs
from .train import rollouts


def pass_at_kt(params, layouts, budget, k=8, rng=None, blind_budget=False):
    """Pass@(K, T): fraction of tasks with >= 1 success among K sampled
    trajectories at budget T. Also returns Pass@(1, T) (mean success rate)."""
    layouts_rep = [l for l in layouts for _ in range(k)]
    budgets = [budget] * len(layouts_rep)
    traj = rollouts(params, layouts_rep, budgets, rng, blind_budget=blind_budget)
    r = traj.rewards.reshape(len(layouts), k)
    return {
        "pass_k": float((r.max(axis=1) > 0).mean()),
        "pass_1": float(r.mean()),
    }


def collect_states(params, layouts, rng, n_states=400, budget=TMAX):
    """Sample intermediate states from the policy's own trajectories on
    held-out layouts. Returns list of (layout, full_state, turn)."""
    states = []
    per_layout = max(1, n_states // len(layouts))
    for layout in layouts:
        env = GridTools(layout)
        for _ in range(per_layout):
            env.reset(rng, budget=budget)
            done = False
            while not done:
                s = env.full_state()
                states.append((layout, s, env.turn))
                obs = env.observe(budget)
                lp = log_probs(params, obs[None, :])[0]
                p = np.exp(lp)
                a = int(rng.choice(N_ACTIONS, p=p / p.sum()))
                _, done = env.step(a, rng)
                if len(states) >= n_states:
                    return states
    return states


def _mean_q_of_sampled_actions(params, layout, Q, s, rng, g=16):
    """Mean Q*(s, a) of G actions sampled from the policy at state s.
    The policy needs an observation; rebuild it from the full state."""
    r, c, has_key, door_open = _state_parts(s)
    env = GridTools(layout)
    env.r, env.c = r, c
    env.has_key, env.door_open = has_key, door_open
    env.turn = 0
    obs = env.observe(TMAX)
    lp = log_probs(params, obs[None, :])[0]
    p = np.exp(lp)
    acts = rng.choice(N_ACTIONS, size=g, p=p / p.sum())
    return float(np.mean([Q[a, s] for a in acts]))


def _state_parts(s):
    from .env import index_to_state

    return index_to_state(s)


def step_quality(base_params, rl_params, layouts, qstars, rng, n_states=400, g=16):
    """Matched-state action quality, paper sec. 3.3: states sampled equally
    from base and RL trajectories; G=16 actions per state scored by exact Q*.
    Returns win rates and mean Q* for each policy."""
    pool = []
    for params in (base_params, rl_params):
        pool.extend(collect_states(params, layouts, rng, n_states=n_states // 2))
    rng.shuffle(pool)
    base_q, rl_q = [], []
    for layout, s, _turn in pool:
        Q = qstars[id(layout)]
        base_q.append(_mean_q_of_sampled_actions(base_params, layout, Q, s, rng, g))
        rl_q.append(_mean_q_of_sampled_actions(rl_params, layout, Q, s, rng, g))
    base_q = np.array(base_q)
    rl_q = np.array(rl_q)
    return {
        "n_states": len(pool),
        "base_mean_q": float(base_q.mean()),
        "rl_mean_q": float(rl_q.mean()),
        "rl_win_rate": float((rl_q > base_q).mean()),
        "base_win_rate": float((base_q > rl_q).mean()),
    }


def _wrong_action(layout, Q, s, rng):
    """Inject an error: take the worst-Q* move action at state s (a genuine
    mistake that moves the agent somewhere it must recover from), and return
    (action, resulting_state). Falls back to a wasted pickup/unlock if every
    move is equally good."""
    from .env import index_to_state

    r, c, has_key, door_open = index_to_state(s)
    env = GridTools(layout)
    env.r, env.c = r, c
    env.has_key, env.door_open = has_key, door_open
    env.turn = 0
    q = Q[:, s]
    worst = min(range(4), key=lambda a: q[a])
    if q[worst] < q.max() - 1e-9:
        env.step(worst, rng)
        return worst, env.full_state()
    # No move is strictly bad here: waste the turn on a failed tool call.
    a = 4 if not (layout.key is not None and (r, c) == layout.key and not has_key) else 5
    env.step(a, rng)
    return a, env.full_state()


def error_state_quality(base_params, rl_params, layouts, qstars, rng,
                        n_states=200, g=16):
    """Paper sec. 3.3 error states: inject one wrong action, then compare
    action quality at the resulting error state, plus continuation success
    from identical error states (K=8 rollouts, T=TMAX)."""
    pool = []
    for params in (base_params, rl_params):
        pool.extend(collect_states(params, layouts, rng, n_states=n_states // 2))
    rng.shuffle(pool)
    base_q, rl_q = [], []
    err_states = []
    for layout, s, _turn in pool:
        Q = qstars[id(layout)]
        _a, s_err = _wrong_action(layout, Q, s, rng)
        err_states.append((layout, s_err))
        base_q.append(_mean_q_of_sampled_actions(base_params, layout, Q, s_err, rng, g))
        rl_q.append(_mean_q_of_sampled_actions(rl_params, layout, Q, s_err, rng, g))
    base_q = np.array(base_q)
    rl_q = np.array(rl_q)
    # Continuation success from identical error states.
    cont = {}
    for name, params in (("base", base_params), ("rl", rl_params)):
        succ = 0
        total = 0
        for layout, s_err in err_states:
            from .env import index_to_state

            r, c, has_key, door_open = index_to_state(s_err)
            for _ in range(8):
                env = GridTools(layout)
                env.reset(rng, budget=TMAX)
                env.r, env.c = r, c
                env.has_key, env.door_open = has_key, door_open
                env.turn = 1  # one turn already spent on the error
                done = False
                while not done:
                    obs = env.observe(TMAX)
                    lp = log_probs(params, obs[None, :])[0]
                    p = np.exp(lp)
                    a = int(rng.choice(N_ACTIONS, p=p / p.sum()))
                    rew, done = env.step(a, rng)
                    if done:
                        succ += int(rew > 0)
                        break
                total += 1
        cont[name] = float(succ / max(total, 1))
    return {
        "n_error_states": len(err_states),
        "base_mean_q": float(base_q.mean()),
        "rl_mean_q": float(rl_q.mean()),
        "rl_win_rate": float((rl_q > base_q).mean()),
        "continuation_base": cont["base"],
        "continuation_rl": cont["rl"],
    }


def depth_quality(base_params, rl_params, layouts, qstars, rng, n_traj=40):
    """Mean Q* of actions taken, by normalized trajectory depth quartile.
    Paper sec. 3.3 / Fig. 6: the RL edge should emerge late, not early."""
    out = {}
    # Stride across the layout list so the sample spans all difficulty tiers
    # (make_layouts groups layouts by tier).
    sampled = layouts[:: max(1, len(layouts) // n_traj)][:n_traj]
    for name, params in (("base", base_params), ("rl", rl_params)):
        buckets = {0: [], 1: [], 2: [], 3: []}
        for layout in sampled:
            env = GridTools(layout)
            env.reset(rng, budget=TMAX)
            Q = qstars[id(layout)]
            done = False
            while not done:
                s = env.full_state()
                q = Q[:, s].max()
                d = min(3, int(4 * env.turn / TMAX))
                obs = env.observe(TMAX)
                lp = log_probs(params, obs[None, :])[0]
                p = np.exp(lp)
                a = int(rng.choice(N_ACTIONS, p=p / p.sum()))
                buckets[d].append(float(Q[a, s] - q))  # regret (<= 0)
                _, done = env.step(a, rng)
        out[name] = {f"q{q}": float(np.mean(v)) if v else None for q, v in buckets.items()}
    return out


def likelihood_analysis(base_params, rl_params, layouts, rng, n_traj=60):
    """Paper sec. 4.1: per-action NLL of RL actions under the base policy, and
    cumulative LLR of successful trajectories under RL vs base."""
    nlls = []
    llr_final = []
    # Stride across the layout list so the sample spans all difficulty tiers.
    sampled = layouts[:: max(1, len(layouts) // n_traj)][:n_traj]
    for layout in sampled:
        for params, tag in ((rl_params, "rl"), (base_params, "base")):
            env = GridTools(layout)
            env.reset(rng, budget=TMAX)
            done = False
            llr = 0.0
            traj_nll = []
            rew = 0.0
            while not done:
                obs = env.observe(TMAX)
                lp_rl = log_probs(rl_params, obs[None, :])[0]
                lp_b = log_probs(base_params, obs[None, :])[0]
                p = np.exp(lp_rl) if tag == "rl" else np.exp(lp_b)
                a = int(rng.choice(N_ACTIONS, p=p / p.sum()))
                if tag == "rl":
                    traj_nll.append(float(-lp_b[a]))
                    llr += float(lp_rl[a] - lp_b[a])
                rew, done = env.step(a, rng)
            if tag == "rl":
                if rew > 0:
                    llr_final.append(llr)
                    nlls.extend(traj_nll)
    return {
        "n_success_traj": len(llr_final),
        "mean_action_nll_under_base": float(np.mean(nlls)) if nlls else None,
        "mean_final_llr_success": float(np.mean(llr_final)) if llr_final else None,
    }
