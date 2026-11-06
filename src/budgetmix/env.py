"""GridTools: a simulated multi-turn tool-use environment.

A 7x7 grid agent that must pick up a key, unlock a door in a barrier wall, and
reach the goal. Six actions (4 moves, pickup, unlock); moves slip
(FrozenLake-style); wrong pickup/unlock wastes a turn with an error
observation; reward 1 only on reaching the goal within the turn budget.

The policy sees a partial observation (3x3 local view + inventory + budget
fractions), so it is a genuine agent policy, not a lookup of the full state.
Exact Q* by value iteration on the full MDP state is available for step-level
evaluation and expert data -- the FrozenLake DP methodology of Zhu et al.
(arXiv:2611.06483, sec. 3.3), applied to our own environment.
"""

from __future__ import annotations

import numpy as np

N = 7  # grid size
N_ACTIONS = 6
UP, DOWN, LEFT, RIGHT, PICKUP, UNLOCK = range(6)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
# Full-MDP state: (row, col, has_key, door_open). 7*7*2*2 = 196 states.
N_STATES = N * N * 2 * 2
TMAX = 60  # full interaction budget

# Budget grid used by BudgetMix (paper Eq. 2, 10% intervals of Tmax).
BUDGET_GRID = tuple(int(np.ceil(k / 10 * TMAX)) for k in range(1, 11))


def state_index(r: int, c: int, has_key: bool, door_open: bool) -> int:
    return ((r * N + c) * 2 + int(has_key)) * 2 + int(door_open)


def index_to_state(idx: int):
    door_open = bool(idx % 2)
    idx //= 2
    has_key = bool(idx % 2)
    idx //= 2
    c = idx % N
    r = idx // N
    return r, c, has_key, door_open


class Layout:
    """One task layout: walls, key, door, goal, start, slip probability."""

    def __init__(self, rng: np.random.Generator, tier: str):
        assert tier in ("easy", "medium", "hard")
        self.tier = tier
        self.slip = {"easy": 0.0, "medium": 0.1, "hard": 0.2}[tier]
        self.walls = np.zeros((N, N), dtype=bool)
        # Border is a wall.
        self.walls[0, :] = True
        self.walls[-1, :] = True
        self.walls[:, 0] = True
        self.walls[:, -1] = True
        self.has_door = tier in ("medium", "hard")
        self.door = None
        self.key = None
        if self.has_door:
            # Vertical barrier at column 3, rows 1..5, with one door cell.
            door_row = int(rng.integers(1, 6))
            for r in range(1, 6):
                if r != door_row:
                    self.walls[r, 3] = True
            self.door = (door_row, 3)
            # Start on the left side, goal on the right side, key on the
            # left side (hard: far from the door, top-left corner region).
            self.start = (int(rng.integers(1, 6)), int(rng.integers(1, 3)))
            self.goal = (int(rng.integers(1, 6)), int(rng.integers(4, 6)))
            if tier == "hard":
                self.key = (1, 1)
            else:
                self.key = (int(rng.integers(1, 6)), int(rng.integers(1, 3)))
                # Key must not coincide with start or the door.
                while self.key == self.start or self.key == self.door:
                    self.key = (int(rng.integers(1, 6)), int(rng.integers(1, 3)))
            while self.goal == self.door:
                self.goal = (int(rng.integers(1, 6)), int(rng.integers(4, 6)))
        else:
            self.start = (int(rng.integers(1, 6)), int(rng.integers(1, 6)))
            self.goal = (int(rng.integers(1, 6)), int(rng.integers(1, 6)))
            while self.goal == self.start:
                self.goal = (int(rng.integers(1, 6)), int(rng.integers(1, 6)))

    def is_wall(self, r: int, c: int) -> bool:
        return self.walls[r, c]

    def is_door(self, r: int, c: int) -> bool:
        return self.door is not None and (r, c) == self.door


def make_layouts(n_per_tier: int, seed: int) -> list[Layout]:
    rng = np.random.default_rng(seed)
    layouts = []
    for tier in ("easy", "medium", "hard"):
        for _ in range(n_per_tier):
            layouts.append(Layout(rng, tier))
    return layouts


class GridTools:
    """Episodic environment. One instance = one layout; reset() starts an episode."""

    def __init__(self, layout: Layout):
        self.layout = layout
        self.r = self.c = 0
        self.has_key = False
        self.door_open = not layout.has_door
        self.turn = 0
        self.done = False
        self.last_error = False
        self.budget = TMAX

    def reset(self, rng: np.random.Generator, budget: int = TMAX) -> None:
        self.r, self.c = self.layout.start
        self.has_key = False
        self.door_open = not self.layout.has_door
        self.turn = 0
        self.done = False
        self.last_error = False
        self.budget = int(budget)

    # -- dynamics ---------------------------------------------------------
    def _move_target(self, action: int, rng: np.random.Generator):
        """Apply slip: intended direction with prob 1-p, else a random other dir."""
        if rng.random() < self.layout.slip:
            others = [d for d in range(4) if d != action]
            action = int(rng.choice(others))
        dr, dc = MOVES[action]
        return self.r + dr, self.c + dc

    def _passable(self, r: int, c: int) -> bool:
        if not (0 <= r < N and 0 <= c < N):
            return False
        if self.layout.is_wall(r, c):
            return False
        if self.layout.is_door(r, c) and not self.door_open:
            return False
        return True

    def step(self, action: int, rng: np.random.Generator):
        """Returns (reward, done). Consumes one turn."""
        assert not self.done
        self.turn += 1
        self.last_error = False
        if action in (UP, DOWN, LEFT, RIGHT):
            nr, nc = self._move_target(action, rng)
            if self._passable(nr, nc):
                self.r, self.c = nr, nc
            # else: bump into a wall/closed door, stay put (turn wasted).
        elif action == PICKUP:
            if self.layout.key is not None and (self.r, self.c) == self.layout.key and not self.has_key:
                self.has_key = True
            else:
                self.last_error = True
        elif action == UNLOCK:
            adj_door = self.layout.door is not None and any(
                (self.r + dr, self.c + dc) == self.layout.door for dr, dc in MOVES
            )
            if adj_door and self.has_key and not self.door_open:
                self.door_open = True
            else:
                self.last_error = True
        else:
            raise ValueError(action)
        if (self.r, self.c) == self.layout.goal:
            self.done = True
            return 1.0, True
        if self.turn >= self.budget:
            self.done = True
            return 0.0, True
        return 0.0, False

    # -- observation -------------------------------------------------------
    def observe(self, budget_total: int, blind_budget: bool = False) -> np.ndarray:
        """Partial observation: coords, 3x3 local view (4 channels), inventory,
        remaining-budget fraction, total-budget fraction. blind_budget=True
        removes the budget cue (extension E2): total fraction is fixed to 1.0
        and the remaining fraction is measured against TMAX."""
        obs = np.zeros(42, dtype=np.float64)
        obs[0] = self.r / (N - 1)
        obs[1] = self.c / (N - 1)
        # Local 3x3 view, channels: wall, key, door(closed), goal. A narrow
        # view is deliberate: the agent cannot see the key/door/goal until
        # close, so it must explore, makes wrong turns, and needs recovery --
        # the error/recovery dynamics the paper studies. A 5x5 view made the
        # BC base near-perfect (98% expert agreement) and saturated Pass@8.
        k = 2
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr, cc = self.r + dr, self.c + dc
                base = k + ((dr + 1) * 3 + (dc + 1)) * 4
                if not (0 <= rr < N and 0 <= cc < N) or self.layout.is_wall(rr, cc):
                    obs[base + 0] = 1.0
                elif self.layout.is_door(rr, cc) and not self.door_open:
                    obs[base + 2] = 1.0
                elif self.layout.key is not None and (rr, cc) == self.layout.key and not self.has_key:
                    obs[base + 1] = 1.0
                elif (rr, cc) == self.layout.goal:
                    obs[base + 3] = 1.0
        obs[38] = float(self.has_key)
        obs[39] = float(self.door_open)
        if blind_budget:
            obs[40] = (TMAX - self.turn) / TMAX
            obs[41] = 1.0
        else:
            obs[40] = (budget_total - self.turn) / max(budget_total, 1)
            obs[41] = budget_total / TMAX
        return obs

    def full_state(self) -> int:
        return state_index(self.r, self.c, self.has_key, self.door_open)


# -- exact Q* by value iteration -------------------------------------------

def build_transitions(layout: Layout):
    """P[a, s, s'] transition probabilities and R[a, s] expected reward."""
    P = np.zeros((N_ACTIONS, N_STATES, N_STATES))
    R = np.zeros((N_ACTIONS, N_STATES))
    for s in range(N_STATES):
        r, c, has_key, door_open = index_to_state(s)
        for a in range(N_ACTIONS):
            if (r, c) == layout.goal:
                P[a, s, s] = 1.0
                continue
            if a in (UP, DOWN, LEFT, RIGHT):
                dests: dict[int, float] = {}
                for d in range(4):
                    p = (1.0 - layout.slip) if d == a else layout.slip / 3.0
                    if p == 0.0:
                        continue
                    dr, dc = MOVES[d]
                    nr, nc = r + dr, c + dc
                    # Passable check mirrors GridTools._passable.
                    ok = 0 <= nr < N and 0 <= nc < N and not layout.is_wall(nr, nc)
                    if layout.is_door(nr, nc) and not door_open:
                        ok = False
                    if not ok:
                        nr, nc = r, c
                    ns = state_index(nr, nc, has_key, door_open)
                    dests[ns] = dests.get(ns, 0.0) + p
                for ns, p in dests.items():
                    P[a, s, ns] = p
                    if index_to_state(ns)[0:2] == layout.goal:
                        R[a, s] += p * 1.0
            elif a == PICKUP:
                if layout.key is not None and (r, c) == layout.key and not has_key:
                    ns = state_index(r, c, True, door_open)
                else:
                    ns = s
                P[a, s, ns] = 1.0
            elif a == UNLOCK:
                adj = layout.door is not None and any(
                    (r + dr, c + dc) == layout.door for dr, dc in MOVES
                )
                if adj and has_key and not door_open:
                    ns = state_index(r, c, has_key, True)
                else:
                    ns = s
                P[a, s, ns] = 1.0
    return P, R


def solve_qstar(layout: Layout, gamma: float = 0.95, tol: float = 1e-8):
    """Value iteration. Returns Q* (6 x 196). Discounted infinite-horizon Q*
    is used as a consistent yardstick for step-level action quality (the
    paper's FrozenLake DP uses gamma=0.95 the same way); it does not depend on
    remaining turns, so base and RL are compared on identical states."""
    P, R = build_transitions(layout)
    V = np.zeros(N_STATES)
    while True:
        Q = R + gamma * np.einsum("asn,n->as", P, V)
        V_new = Q.max(axis=0)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q = R + gamma * np.einsum("asn,n->as", P, V)
    return Q


def expert_action(Q: np.ndarray, s: int, rng: np.random.Generator) -> int:
    """Greedy action w.r.t. Q*, ties broken uniformly at random."""
    q = Q[:, s]
    best = np.flatnonzero(q == q.max())
    return int(rng.choice(best))
