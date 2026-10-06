"""One run = train a DQN agent under one condition, then record a long evaluation life."""
import copy
import json
import os

import numpy as np
import torch

from .dqn import DQNAgent
from .env import N_ACTIONS, Interoception, InteroceptiveGridWorld
from .rewards import make_reward


def run_one(cfg, out_path=None):
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    env = InteroceptiveGridWorld(cfg.env, rng)
    intero = Interoception(cfg.reward.sigma, rng)
    reward_fn = make_reward(cfg.reward, cfg.env)
    agent = DQNAgent(env.obs_dim, N_ACTIONS, cfg.agent, rng)
    setpoint = env.setpoint

    def new_life():
        env.reset()
        reward_fn.reset()
        h_hat = intero(env.h)
        return h_hat, env.observation(h_hat)

    # ---- training ---------------------------------------------------------
    n_blocks = cfg.train_steps // cfg.log_every
    train_log = {k: np.zeros((n_blocks,) + s) for k, s in [
        ("intake", (cfg.env.n_vars,)), ("deaths", ()), ("abs_dev", ()), ("reward", ())]}
    h_hat, obs = new_life()
    for step in range(cfg.train_steps):
        a = agent.act(obs, agent.epsilon(step))
        consumed, dead, truncated = env.step(a)
        h_hat_next = intero(env.h)
        r = reward_fn(h_hat, h_hat_next, consumed) - dead * cfg.reward.death_penalty
        obs_next = env.observation(h_hat_next)
        agent.buffer.add(obs, a, r, obs_next, float(dead))
        agent.update(step)

        b = step // cfg.log_every
        if b < n_blocks:
            train_log["intake"][b] += consumed
            train_log["deaths"][b] += dead
            train_log["abs_dev"][b] += np.abs(env.h - setpoint).mean()
            train_log["reward"][b] += r

        if dead or truncated:
            h_hat, obs = new_life()
        else:
            h_hat, obs = h_hat_next, obs_next
    for k in ("intake", "abs_dev", "reward"):
        train_log[k] /= cfg.log_every

    # ---- evaluation: one continuous life, reset only on death -------------
    env.max_steps = cfg.eval_steps + 1
    T = cfg.eval_steps
    trace = {
        "h": np.zeros((T, cfg.env.n_vars)), "h_hat": np.zeros((T, cfg.env.n_vars)),
        "consumed": np.zeros((T, cfg.env.n_vars)), "pos": np.zeros((T, 2), np.int64),
        "dead": np.zeros(T, bool), "warning": np.zeros(T, bool), "scarcity": np.zeros(T, bool),
    }
    h_hat, obs = new_life()
    for t in range(T):
        trace["warning"][t], trace["scarcity"][t] = env.in_warning, env.in_scarcity
        a = agent.act(obs, cfg.eval_eps)
        consumed, dead, _ = env.step(a)
        h_hat_next = intero(env.h)
        trace["h"][t], trace["h_hat"][t], trace["consumed"][t] = env.h, h_hat_next, consumed
        trace["pos"][t], trace["dead"][t] = env.pos, dead
        if dead:
            h_hat, obs = new_life()
        else:
            h_hat, obs = h_hat_next, env.observation(h_hat_next)

    result = {**{f"train_{k}": v for k, v in train_log.items()}, **trace}
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = out_path + ".tmp.npz"
        np.savez_compressed(tmp, config=json.dumps(cfg.to_dict()), **result)
        os.replace(tmp, out_path)
    return result


def with_overrides(cfg, **sections):
    """Copy a RunConfig, overriding fields per section: with_overrides(cfg, reward={'sigma': 1})."""
    cfg = copy.deepcopy(cfg)
    for section, values in sections.items():
        target = cfg if section == "run" else getattr(cfg, section)
        for k, v in values.items():
            setattr(target, k, v)
    return cfg
