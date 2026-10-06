"""Reward functions for the four conditions.

All share the signature reward(h_hat, h_hat_next, consumed) so the agent and the
training loop are identical across conditions: only the reward changes.
"""
import math
from collections import deque

import numpy as np


def drive(h, setpoint, n, m):
    """Keramati & Gutkin (2014): D(H) = (sum_i |h*_i - h_i|^n)^(1/m), n > m > 1.
    Symmetric, so overshooting the setpoint is penalised like a deficit."""
    return float((np.abs(setpoint - h) ** n).sum() ** (1.0 / m))


class HomeostaticReward:
    """HRRL: reward = drive reduction, r_t = D(h_hat_t) - D(h_hat_{t+1})."""

    def __init__(self, setpoint, n, m):
        self.setpoint, self.n, self.m = setpoint, n, m

    def reset(self):
        pass

    def __call__(self, h_hat, h_hat_next, consumed):
        return (drive(h_hat, self.setpoint, self.n, self.m)
                - drive(h_hat_next, self.setpoint, self.n, self.m))


class MaximizerReward:
    """Reward = quantity consumed. No internal bound at all."""

    def reset(self):
        pass

    def __call__(self, h_hat, h_hat_next, consumed):
        return float(consumed.sum())


class SaturatingReward:
    """Hand-coded saturation: a bite of variable i is rewarded (+1) only if fewer than
    cap_i bites of i occurred in the last `window` steps. Bounded, but blind to the
    internal state, so it cannot produce state-dependent resumption."""

    def __init__(self, caps, window):
        self.caps, self.window = caps, window
        self.reset()

    def reset(self):
        self.t = 0
        self.history = [deque() for _ in self.caps]

    def __call__(self, h_hat, h_hat_next, consumed):
        self.t += 1
        r = 0.0
        for i, amount in enumerate(consumed):
            hist = self.history[i]
            while hist and hist[0] <= self.t - self.window:
                hist.popleft()
            if amount > 0:
                if len(hist) < self.caps[i]:
                    r += 1.0
                hist.append(self.t)
        return r


def matched_caps(env_cfg, window):
    """Bites per window needed to offset metabolism while commuting between resources."""
    need = (np.asarray(env_cfg.basal_cost) + np.asarray(env_cfg.move_cost)) * window / env_cfg.bite
    return tuple(int(math.ceil(x)) for x in need)


def make_reward(reward_cfg, env_cfg):
    setpoint = np.full(env_cfg.n_vars, env_cfg.setpoint)
    if reward_cfg.condition == "hrrl":
        return HomeostaticReward(setpoint, reward_cfg.drive_n, reward_cfg.drive_m)
    if reward_cfg.condition == "maximizer":
        return MaximizerReward()
    if reward_cfg.condition == "saturating":
        caps = reward_cfg.sat_cap or matched_caps(env_cfg, reward_cfg.sat_window)
        return SaturatingReward(caps, reward_cfg.sat_window)
    raise ValueError(f"unknown condition {reward_cfg.condition!r}")
