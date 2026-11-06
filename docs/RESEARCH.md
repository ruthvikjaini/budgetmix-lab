# Research notes — budgetmix-lab (Day 13, 2026-11-06)

## Trending-first check (2026-11-06, ~00:00 ET)

GitHub daily/weekly trending, same pattern as the last eight nights: skill/prompt
packs dominate (claude-mem ~108K stars, superpowers, impeccable, hallmark,
security-audit-skill, anthropics/skills), plus full platforms and apps
(Orkas-AI/Orkas 7,016 stars / 439 forks, local-first multi-agent desktop app;
THU-MAIC/OpenMAIC 42,318 stars / 6,522 forks, AI classroom; DeepTutor ~53K;
agent-desktop; ghostty; TradingAgents). Papers-with-code velocity list is led by
google-research/timesfm (35K stars, +392/week — a time-series foundation model,
needs the pretrained checkpoint) and small utilities (textfold, local-deep-research).
Examined and rejected as optimizable targets:

- Orkas / OpenMAIC / DeepTutor / TradingAgents: full applications/platforms, no
  single benchmarkable code weakness a one-night CPU build could measure.
- JustVugg/colibri, FlashML-org/FreeToken: MoE inference engines, GPU-only.
- kimi-k3-in-c (10,443 stars, 1,643 forks): needs the 1.56 TB checkpoint.
- NVIDIA/auto-ontology (5,526 stars, 436 forks): enterprise stack needing NIM keys.
- patchy631/ai-engineering-hub (40K stars): tutorial collection, not code to optimize.
- kornia: mature differentiable-CV library, no one-night measurable weakness.

No trending repo picked; paper fallback per the standing rule.

## Paper chosen

Zhu, Jin, Tu, Men, Hao, Li, Liu & Zhao, "What Does Multi-Turn Reinforcement
Learning Really Improve in LLM Agents?", arXiv:2611.06483, submitted 4 Nov 2026.
Verified on the arXiv abstract page and full HTML, 2026-11-06. Project page:
rethinking-agentic-rl.github.io. No paper code read or used; independent
implementation from the paper's equations and tables.

### What the paper shows (paper numbers, kept separate from our measurements)

- Metric: Pass@(K,T) — sampling breadth K x interaction depth T. Fixing K=8 and
  sweeping T isolates the depth axis (Pass@T).
- RL gains depend on budget. Qwen3-4B, GRPO-Full vs base, Pass@8 at 25% of Tmax:
  FrozenLake 28.7 vs 30.3 (RL *worse*), DeepCoder-MT 29.2 vs 34.0 (worse),
  BFCL-v3 2.5 vs 3.0 (worse). At 100%: 72.4 vs 61.6, 59.0 vs 48.3, 30.2 vs 26.4.
  The advantage appears only once the budget is ample.
- Step-level action quality does NOT uniformly improve. On matched states, RL
  picks the better action on only 46.8% of DeepCoder-MT states and 42.8% of
  FrozenLake states (exact Q* via DP on FrozenLake, unit-test scores on
  DeepCoder-MT); average quality slightly below base.
- RL excels *after errors*: on error states (an incorrect action injected at the
  previous step) RL picks the better action 65.7% (FrozenLake) / 67.3%
  (DeepCoder-MT) of the time; continuation success from identical error states:
  FrozenLake 51.3% vs 38.6%, DeepCoder-MT 38.5% vs 22.9%.
- Advantage emerges late in trajectories: little/no edge early, gap reverses as
  interaction unfolds (normalized depth d_t = t/Tmax).
- Mechanism (§4.1): per-action NLL under the base policy is similar for RL and
  base actions (means 12.20 vs 10.25 on Terminal-Bench) — RL actions stay locally
  plausible — but cumulative LLR of successful RL trajectories reaches ~228 vs
  ~26 for base trajectories. RL composes locally plausible actions into reliable
  long trajectories; it does not widen the per-step frontier.
- Training dynamics (§4.2, DeepCoder-MT): trajectory reward R_multi rises +0.331
  while step reward R_single *falls* -0.097. The step-quality decline emerges
  progressively, not just at the final checkpoint.
- Mitigations (§5): oracle Q-value process rewards / GiGPO stabilize Pass@T on
  FrozenLake, but on BFCL (no exact step oracle) dense methods fail to help under
  tight budgets — precise process supervision is rarely available in practice.
- BudgetMix RL (§5.2): one modification — sample the training budget per task
  group uniformly from T(x) = Unique({ceil(k/10 * Tmax(x)) : k = 1..10}); the total
  and remaining budget are given in the prompt; trajectories exceeding the budget
  score 0. Same reward as standard RL. Gains over full-budget RL (Qwen3-4B,
  averaged over GRPO/GiGPO/PPO): +7.5pp at 25%, +5.3pp at 50%, +4.0pp at 75%,
  parity at 100% (+1.1 to +1.6pp). Overall +4.2pp (4B) / +4.8pp (8B). During
  BudgetMix training R_single stays stable or improves slightly while R_multi
  rises — it mitigates the step-quality degradation.
- Benchmarks: FrozenLake (Tmax=70), DeepCoder-MT (10), BFCL-v3 (24),
  InterCode-SQL (10). Models Qwen3-4B / 8B. Algorithms GRPO, GiGPO, PPO.

### Other candidates considered / rejected

- Matrenok et al., "Controlling Sharpening in LLM Post-Training" (KAPPO),
  arXiv:2611.03146, 2 Nov 2026 — closed-form optimal policies for KL-regularized
  pass@k / max@k; prompt-adaptive regularization. Strong paper, but its mechanism
  (pass@1 vs pass@k coverage trade-off) sits in the same cluster as Liu's
  No-Free-Lunch RLVR paper and our Days 1/7/10/12 single-turn RLVR builds.
  Held for a later night.
- Liu, "There Is No Free Lunch in RLVR" (arXiv:2610.39295, 29 Oct 2026) — held a
  fourth night; same-cluster reason as above.
- Chen et al., BLPO (arXiv:2611.03005, 2 Nov 2026) — cross-task advantages for
  long-horizon agents; needs ALFWorld/WebShop-scale rollouts to reproduce
  honestly. Noted, not built.
- Zhang et al., proactive elicitation / DialogGRPO (arXiv:2611.01878, 1 Nov 2026)
  — VagueBench + dialogue-state DAG; the evaluation needs a dialogue simulator
  and a judge, too much unverifiable machinery for one CPU night.
- Jia et al., Axion 1.0 (arXiv:2611.03233) — 35B/122B agent foundation model;
  GPU training, not CPU-reproducible.
- Zhou et al., "Route Anything" (arXiv:2611.05275, 3 Nov 2026) — routing gives
  8.82x efficiency across 25 benchmarks / 374K runs; reproducing the routing
  claim needs the benchmark suite, not a one-night CPU build.

## This build: budgetmix-lab

An exact, CPU-only reproduction/extension of Zhu et al. on a simulated
multi-turn tool-use testbed, plus two extensions the paper does not run.

Testbed ("GridTools", our own environment, no paper code): a 7x7 grid agent that
must pick up a key, unlock a door in a barrier wall, and reach the goal. Six
actions (4 moves, pickup, unlock); slip probability makes moves stochastic
(FrozenLake-style); wrong pickup/unlock wastes a turn with an error observation;
reward 1 only on reaching the goal within the turn budget, else 0. Partial
observation (3x3 local view + inventory + budget fractions), so the policy is a
genuine agent policy, not a DP lookup. Three difficulty tiers (slip 0 / 0.1 / 0.2,
door on/off, key placement). Exact Q* by value iteration on the full MDP state
is available for step-quality evaluation and expert data — the paper's FrozenLake
DP methodology, applied to our own environment.

Policy: 2-layer MLP (NumPy, manual backprop, gradient-checked vs finite
differences), trained by behavior cloning on DP-optimal trajectories to form the
base checkpoint, then GRPO from that same base for both arms:

- GRPO-Full: every training group runs at T = Tmax = 60 (paper's standard RL).
- GRPO-BudgetMix: per task group, T ~ Uniform({6, 12, 18, 24, 30, 36, 42, 48, 54,
  60}) — the paper's Eq. (2) 10% grid; budget fractions are policy inputs,
  exactly as the paper puts total/remaining budget in the prompt.

Experiments (all seeded, held-out task set, 3 training seeds unless noted):

1. Pass@(K=8, T) grid at T = 15/30/45/60 (25/50/75/100% of Tmax) for base,
   GRPO-Full, GRPO-BudgetMix — the paper's Table 1 on our testbed.
2. Step-level quality: matched-state pool (states sampled equally from base and
   RL trajectories), G=16 actions per state scored by exact Q*; win rates and
   mean Q* gap, on all states and on error states (wrong action injected at the
   previous step), plus continuation success from identical error states.
   Also action quality vs normalized trajectory depth.
3. Training dynamics: R_multi (trajectory reward on a fixed validation set) and
   R_single (mean Q* of actions taken) per checkpoint — the paper's Fig. 8.
4. Likelihood: per-action NLL of RL actions under the base policy; cumulative
   LLR of successful trajectories under RL vs base — the paper's Fig. 7.
5. Extensions (ours, not in the paper):
   E1. Unseen-budget interpolation: evaluate at T = 9/21/33/51 (15/35/55/85%),
   which neither method trained on (BudgetMix's grid is multiples of 6).
   Does BudgetMix interpolate, or only memorize its training grid?
   E2. Budget-blind evaluation: the same policies evaluated with the budget cue
   removed (total-budget input fixed to 1.0, remaining fraction measured against
   Tmax). In deployment the agent often does not know its budget; is BudgetMix's
   robustness cue-dependent or genuine prudence?

All numbers in results/ are simulation numbers on this testbed, not paper
numbers; paper numbers are quoted above and in the README only for comparison.

## Bugs found and fixed during the night

1. `GridTools.step` referenced `self.budget` before it existed (AttributeError on
   the first rollout) — fixed by adding a `budget` parameter to `reset()`.
2. First pilot: 5x5 local view made the BC base near-perfect (98.1% expert
   agreement, Pass@8 = 1.000 at T=40 on 12 layouts) and saturated every
   comparison. Narrowed the view to 3x3; agreement fell to ~0.90 and the
   budget ladder discriminates (pilot 2: base Pass@8 0.775 at T=15).
3. First pilot used TMAX=40: at T=10 many tasks were structurally
   near-impossible (optimal paths are 12–18 steps), so gains were compressed
   into noise. Raised TMAX to 60 (25% budget = 15 turns, tight but feasible;
   ratio to optimal path ~4x, as in the paper's FrozenLake, Tmax=70).
4. `_wrong_action` first picked a move into a wall, but slip could redirect it
   into a real move — the "error state" was not an error. Redefined to the
   worst-Q* move action (a genuine mistake the agent must recover from).
5. `run_grid.aggregate` computed the Pass@ table inside its own recursion, so
   the nested call looked up `pass_*` keys in a list of pass dicts (KeyError).
   Moved the table computation to the top level.
6. `depth_quality` and `likelihood_analysis` sampled `layouts[:n]`, which is
   all easy-tier layouts (`make_layouts` groups by tier). Changed to strided
   sampling across the list.
7. `test_branding` flagged its own compiled bytecode (the test lists the
   forbidden tokens). The scan now skips `__pycache__` / `.pyc` and itself.
8. Training-dynamics logging first ran at TMAX, where the validation set
   saturates at R_multi ≈ 0.99 from step 0 and the curves carry no signal.
   Dynamics are now measured at T=15 (25% budget), where the base has
   headroom; documented in `eval_dynamics`.

## Pilot results (seed 0, superseded by the full 3-seed grid)

Pilot 1 (TMAX=40, 200 steps, 120 eval layouts): base Pass@8 0.458/0.942/0.983/
1.000 at T=10/20/30/40; full 0.500; mix 0.517 at T=10. Step-quality win rates
already showed the paper's pattern (RL 25% vs base 48%), and error-state
continuation was 0.983 (RL) vs 0.81 (base). Discarded as the design changed
(TMAX, view, eval size); numbers are not quoted in the README.
Pilot 2 (TMAX=60, 250 steps, 240 eval layouts): base 0.775/0.963/0.996/1.000,
full 0.812/0.988/1.000/1.000, mix 0.842/0.996/1.000/1.000 at T=15/30/45/60;
Pass@1 at T=15: 0.514/0.537/0.588. Same qualitative patterns; full grid
re-runs all three seeds with the dynamics fix above.
