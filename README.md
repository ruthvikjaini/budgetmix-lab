# budgetmix-lab

Reproduction and extension of Zhu et al., "What Does Multi-Turn Reinforcement
Learning Really Improve in LLM Agents?" (arXiv:2611.06483, 4 Nov 2026):
multi-turn RL gains depend on the interaction budget, and BudgetMix RL —
training across sampled interaction budgets — restores robust performance at
tight budgets. Built on a small, CPU-only, simulated multi-turn tool-use
testbed. No LLM calls, no GPU, NumPy only.

## The paper's claims (paper numbers, for comparison)

- `Pass@(K,T)` — sampling breadth K x interaction depth T. Fixing K=8 and
  sweeping T, GRPO-Full beats the base model only once the budget is ample;
  at 25% of Tmax it can score *below* the base (Qwen3-4B: FrozenLake 28.7 vs
  30.3, DeepCoder-MT 29.2 vs 34.0, BFCL-v3 2.5 vs 3.0).
- Step-level action quality does **not** uniformly improve: RL picks the
  better action on only 46.8% (DeepCoder-MT) / 42.8% (FrozenLake) of matched
  states. But on error states it wins 65.7% / 67.3%, and continuation success
  from identical error states is 51.3% vs 38.6% (FrozenLake).
- Mechanism: per-action NLL under the base policy is similar for RL and base
  actions (means 12.20 vs 10.25), but cumulative LLR of successful RL
  trajectories reaches ~228 vs ~26 for base. RL composes locally plausible
  actions into reliable long trajectories.
- Training dynamics: trajectory reward R_multi rises +0.331 while step reward
  R_single *falls* -0.097.
- BudgetMix RL (Eq. 2): sample the training budget per task group uniformly
  from `{ceil(k/10 * Tmax) : k = 1..10}`; total and remaining budget are given
  in the prompt; exceeding the budget scores 0. Gains over full-budget RL:
  +7.5pp at 25%, +5.3pp at 50%, +4.0pp at 75%, parity at 100% (average over
  GRPO/GiGPO/PPO, Qwen3-4B). During BudgetMix training R_single stays stable
  or improves slightly.

## This testbed

**GridTools** (our own environment, no paper code): a 7x7 grid agent that must
pick up a key, unlock a door in a barrier wall, and reach the goal. Six
actions (4 moves, pickup, unlock); moves slip with probability 0 / 0.1 / 0.2
by difficulty tier (FrozenLake-style); wrong pickup/unlock wastes a turn with
an error observation; reward 1 only on reaching the goal within the turn
budget, else 0. The policy sees a partial observation (3x3 local view +
inventory + total/remaining budget fractions), so it is a genuine agent
policy, not a lookup of the full state. Exact Q\* by value iteration on the
full MDP state is available for step-level evaluation and expert data — the
paper's FrozenLake DP methodology, applied to our own environment.

**Policy:** 2-layer MLP (42 -> 64 tanh -> 6), NumPy, manual backprop,
gradient-checked against finite differences. Trained by behavior cloning on
DP-optimal trajectories to form the base checkpoint (expert agreement 0.897),
then GRPO from that same base for both arms:

- **GRPO-Full:** every training group runs at T = Tmax = 60 (paper's standard RL).
- **GRPO-BudgetMix:** per task group, T ~ Uniform({6, 12, 18, 24, 30, 36, 42,
  48, 54, 60}) — the paper's Eq. (2) 10% grid.

GRPO: group G=8 rollouts per task, group-normalized advantages, PPO-clipped
surrogate (eps 0.2), KL penalty beta 0.02 to the base checkpoint, Adam lr
3e-4, 250 steps, 24 tasks/step. 3 training seeds; 90 train layouts, 24
validation layouts, 240 held-out layouts (80 per tier).

## Results (all simulation numbers on this testbed, means over 3 seeds)

### Pass@(8, T) — the paper's Table 1

| method | T=15 (25%) | T=30 (50%) | T=45 (75%) | T=60 (100%) |
|---|---|---|---|---|
| base | 0.806 | 0.976 | 0.997 | 1.000 |
| GRPO-Full | 0.828 | 0.987 | 0.999 | 1.000 |
| GRPO-BudgetMix | 0.856 | 0.994 | 1.000 | 1.000 |

BudgetMix over Full: +0.028 at 25%, +0.007 at 50%, +0.001 at 75%, +0.000
at 100%. The shape matches the paper (gains concentrated at tight budgets,
parity at full budget) at smaller magnitude — this testbed saturates at
75–100%, so the absolute headroom is smaller than the paper's benchmarks.
BudgetMix beats Full at 25% on 2 of 3 seeds (per-seed: base 0.775/0.817/
0.825, Full 0.779/0.883/0.821, BudgetMix 0.842/0.871/0.854).

### Pass@(1, T) — single-sample accuracy shows the effect more clearly

| method | T=15 | T=30 | T=45 | T=60 |
|---|---|---|---|---|
| base | 0.522 | 0.831 | 0.933 | 0.967 |
| GRPO-Full | 0.545 | 0.872 | 0.961 | 0.984 |
| GRPO-BudgetMix | 0.593 | 0.930 | 0.969 | 0.973 |

BudgetMix over Full (Pass@1): +0.048 at 25%, +0.058 at 50%, +0.008 at 75%,
-0.012 at 100%. The low-budget robustness costs a little single-sample
accuracy at full budget — reported as measured, not hidden.

### What RL does *not* improve (paper sec. 3.3, reproduced)

On 400 matched states (sampled equally from base and RL trajectories),
G=16 actions per state scored by exact Q\*:

| comparison | RL mean Q\* | base mean Q\* | RL win rate | base win rate |
|---|---|---|---|---|
| Full vs base | 0.844 | 0.846 | 0.310 | 0.393 |
| BudgetMix vs base | 0.826 | 0.836 | 0.219 | 0.508 |

RL does not uniformly improve per-step action quality — same conclusion as
the paper (RL wins 46.8% / 42.8% of matched states there; 31.0% / 21.9%
here), even though trajectory success improves.

### Error recovery (paper sec. 3.3, reproduced)

On 200 error states (a worst-Q\* action injected at the previous step):

| comparison | RL win rate (Q\*) | continuation, RL | continuation, base |
|---|---|---|---|
| Full vs base | 0.328 | 0.998 | 0.956 |
| BudgetMix vs base | 0.347 | 0.994 | 0.943 |

Per-step Q\* at error states still favors the base, but continuation success
from identical error states favors RL by +4.2pp / +5.1pp — the recovery
advantage lives in trajectory composition, not in single-step quality.

### Training dynamics (paper Fig. 8, at T=15 where the base has headroom)

| arm | R_multi step 0 -> 250 | R_single step 0 -> 250 | regret step 0 -> 250 |
|---|---|---|---|
| Full | 0.486 -> 0.559 | 0.718 -> 0.717 | 0.016 -> 0.014 |
| BudgetMix | 0.486 -> 0.590 (peak 0.646 at step 175) | 0.718 -> 0.719 | 0.016 -> 0.012 |

Trajectory reward rises for both arms while step reward stays flat for
full-budget RL — the paper's pattern. BudgetMix additionally *improves* step
regret (0.016 -> 0.012), i.e. it mitigates the step-quality degradation.

### Likelihood (paper sec. 4.1)

Mean per-action NLL of RL actions under the base policy: 0.384 (Full),
0.444 (BudgetMix) — RL actions stay locally plausible under the base.
Mean cumulative LLR of successful trajectories under RL vs base: 0.511
(Full), 1.760 (BudgetMix), positive as in the paper (their scale is larger
because it sums over token sequences; ours sums over ~15 actions).

### Depth profile

Action regret by normalized trajectory-depth quartile is thin: most episodes
end by turn ~20 of 60, so the last quartile is largely unpopulated and the
paper's "advantage emerges late" check is only partially supported here. In
the populated middle quartile (q2), RL regret is lower than base in 2 of 3
seeds for Full. No strong claim is made from this table.

## Extensions (ours, not in the paper)

### E1. Unseen-budget interpolation — Pass@8 at budgets neither method trained on

| method | T=9 | T=21 | T=33 | T=51 |
|---|---|---|---|---|
| base | 0.399 | 0.942 | 0.990 | 0.999 |
| GRPO-Full | 0.408 | 0.949 | 0.996 | 1.000 |
| GRPO-BudgetMix | 0.439 | 0.972 | 0.992 | 1.000 |

BudgetMix is still best at the two tight unseen budgets (+0.031 over Full
at T=9, +0.024 at T=21) and at parity above that. The gain is interpolation
across budgets, not memorization of the training grid.

### E2. Budget-blind evaluation — is the robustness in the cue or the policy?

Same policies evaluated with the budget cue removed (total-budget input
fixed to 1.0, remaining fraction measured against Tmax), Pass@8 at T=15:

| method | cued | blind |
|---|---|---|
| base | 0.783 | 0.854 |
| GRPO-Full | 0.821 | 0.871 |
| GRPO-BudgetMix | 0.856 | 0.842 |

The BudgetMix advantage at T=15 **disappears** when the cue is removed —
and base/Full actually do *better* blind than cued. The reason is visible
in the training distributions: Full was trained only with total-budget
fraction 1.0, which is exactly the blind input, so cued evaluation at T=15
(total fraction 0.25, never seen in training) is a distribution shift for
it; BudgetMix trained across total fractions 0.1–1.0, so the cued input is
in-distribution and the blind input is slightly out of it. Conclusion:
on this testbed, BudgetMix's tight-budget robustness is cue-conditional —
it learns to *use* the budget cue, not cue-independent prudence. That
qualifies the paper's claim in a deployment-relevant way: if the deployed
agent is not told its budget, the BudgetMix edge at tight budgets does not
transfer. At T=60 all methods are 1.000 blind and cued (BudgetMix cued
0.999), so the finding is specific to tight budgets.

## What did not reproduce

- **The below-base reversal at 25%:** the paper's GRPO-Full scores below the
  base at 25% on 3 of 4 benchmarks. Here Full beats base at every budget
  (Pass@8 +0.022 at 25%, +0.011 at 50%). The reversal is benchmark-dependent,
  not universal.
- **Magnitude of the BudgetMix gain:** +0.028 Pass@8 at 25% here vs +0.075
  in the paper — the testbed saturates at high budgets and the base is
  stronger (BC on exact DP), so absolute headroom is smaller.
- **Depth profile:** largely unpopulated in the last quartile (see above).

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q            # 19 tests
python demo.py                  # tiny end-to-end run (~15 s)
python scripts/run_grid.py      # full 3-seed grid (~9 min, results/grid.json)
python scripts/run_extension.py # E1 + E2 from checkpoints (results/extension.json)
python scripts/crossfoot.py     # recompute every README figure from results/
```

Results live in `results/`: `grid.json` (per-seed), `measured.json`
(aggregated), `extension.json`, and `checkpoints/` (base/full/budgetmix
parameters per seed, `.npz`). Reproduce any number with
`scripts/crossfoot.py`, which fails loudly on any mismatch.

## Sources

- Zhu, Jin, Tu, Men, Hao, Li, Liu & Zhao, "What Does Multi-Turn
  Reinforcement Learning Really Improve in LLM Agents?", arXiv:2611.06483,
  4 Nov 2026. https://arxiv.org/abs/2611.06483
- GitHub trending check, 6 Nov 2026 (see `docs/RESEARCH.md`): daily/weekly
  trending dominated by skill/prompt packs and full agent platforms; no
  trending repo had a benchmarkable one-night CPU weakness, so the build
  falls back to the paper above per the standing rule.
- Method references: Schulman et al. (PPO, 2017); Guo et al. (GRPO, 2025);
  Yue et al., "Does RL Really... " (Pass@K capability boundary, 2025).

## License

MIT. See `LICENSE`. Independent implementation from the paper's equations and
tables; no paper code was read or used.
