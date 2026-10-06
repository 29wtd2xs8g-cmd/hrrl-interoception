"""Gridworld with inexhaustible resources and a multi-dimensional internal state.

The true internal state h decays every step (basal metabolism) and faster when the
agent moves. Consuming on a resource cell adds `bite` to the variable that resource
feeds. The agent dies when any variable reaches 0. Moving to food costs water and
vice versa, which creates the circle of necessity the agent must adapt to.

The environment only knows the *true* state. What the agent perceives is produced
by `Interoception`, so noise never leaks into physiology.
"""
import numpy as np

MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # up, down, left, right
STAY, CONSUME = 4, 5
N_ACTIONS = 6


class Interoception:
    """Perceived internal state: h_hat = h + N(0, sigma^2), sampled once per time step.

    The same sample feeds the policy input and the reward, i.e. noise enters both
    the observation and the drive computation."""

    def __init__(self, sigma, rng):
        self.sigma = sigma
        self.rng = rng

    def __call__(self, h):
        if self.sigma == 0:
            return h.copy()
        return h + self.rng.normal(0.0, self.sigma, size=h.shape)


class InteroceptiveGridWorld:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self.setpoint = np.full(cfg.n_vars, cfg.setpoint)
        self.basal = np.asarray(cfg.basal_cost, dtype=float)
        self.move_cost = np.asarray(cfg.move_cost, dtype=float)
        self.resource_at = {(r, c): v for r, c, v in cfg.resources}
        self.max_steps = cfg.max_steps
        self.obs_dim = cfg.size * cfg.size + cfg.n_vars + (3 if cfg.scarcity else 0)

    # ---- episode control -------------------------------------------------
    def reset(self):
        cfg = self.cfg
        self.pos = self.rng.integers(0, cfg.size, size=2)
        self.h = self.rng.uniform(cfg.init_low, cfg.init_high, size=cfg.n_vars)
        self.t = 0
        if cfg.scarcity:
            self.scarcity_start = cfg.scarcity_warning + self._gap()
            self.scarcity_end = self.scarcity_start + cfg.scarcity_duration

    def _gap(self):
        lo, hi = self.cfg.scarcity_interval
        return int(self.rng.integers(lo, hi + 1))

    # ---- scarcity schedule -----------------------------------------------
    @property
    def in_scarcity(self):
        return self.cfg.scarcity and self.scarcity_start <= self.t < self.scarcity_end

    @property
    def in_warning(self):
        return (self.cfg.scarcity
                and self.scarcity_start - self.cfg.scarcity_warning <= self.t < self.scarcity_start)

    def _scarcity_features(self):
        cfg = self.cfg
        warning = 0.0
        if self.in_warning and cfg.scarcity_cue:
            warning = 1.0 - (self.scarcity_start - self.t) / cfg.scarcity_warning
        remaining = ((self.scarcity_end - self.t) / cfg.scarcity_duration) if self.in_scarcity else 0.0
        return [warning, float(self.in_scarcity), remaining]

    # ---- dynamics ----------------------------------------------------------
    def observation(self, h_perceived):
        cfg = self.cfg
        pos = np.zeros(cfg.size * cfg.size)
        pos[self.pos[0] * cfg.size + self.pos[1]] = 1.0
        parts = [pos, (h_perceived - self.setpoint) / self.setpoint]
        if cfg.scarcity:
            parts.append(self._scarcity_features())
        return np.concatenate(parts).astype(np.float32)

    def step(self, action):
        """Returns (consumed, dead, truncated). `consumed` is the nominal amount
        ingested per variable this step, before the h_max ceiling."""
        cfg = self.cfg
        consumed = np.zeros(cfg.n_vars)
        cost = self.basal.copy()
        if action < 4:
            self.pos = np.clip(self.pos + MOVES[action], 0, cfg.size - 1)
            cost += self.move_cost
        elif action == CONSUME:
            var = self.resource_at.get((int(self.pos[0]), int(self.pos[1])))
            if var is not None and not self.in_scarcity:
                consumed[var] = cfg.bite

        self.h = np.minimum(self.h - cost + consumed, cfg.h_max)
        self.t += 1
        if cfg.scarcity and self.t >= self.scarcity_end:
            self.scarcity_start = self.scarcity_end + self._gap() + cfg.scarcity_warning
            self.scarcity_end = self.scarcity_start + cfg.scarcity_duration

        dead = bool((self.h <= 0).any())
        truncated = (not dead) and self.t >= self.max_steps
        return consumed, dead, truncated
