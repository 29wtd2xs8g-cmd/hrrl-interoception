"""Double DQN with a uniform replay buffer."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, obs_dim, size, rng):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.next_obs = np.zeros((size, obs_dim), np.float32)
        self.act = np.zeros(size, np.int64)
        self.rew = np.zeros(size, np.float32)
        self.done = np.zeros(size, np.float32)
        self.size, self.ptr, self.full = size, 0, False
        self.rng = rng

    def add(self, o, a, r, o2, d):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i], self.next_obs[i], self.done[i] = o, a, r, o2, d
        self.ptr = (i + 1) % self.size
        self.full = self.full or self.ptr == 0

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, n):
        idx = self.rng.integers(0, len(self), size=n)
        return (torch.from_numpy(self.obs[idx]), torch.from_numpy(self.act[idx]),
                torch.from_numpy(self.rew[idx]), torch.from_numpy(self.next_obs[idx]),
                torch.from_numpy(self.done[idx]))


class DQNAgent:
    def __init__(self, obs_dim, n_actions, cfg, rng):
        self.cfg, self.rng, self.n_actions = cfg, rng, n_actions
        self.q = QNetwork(obs_dim, n_actions, cfg.hidden)
        self.q_target = QNetwork(obs_dim, n_actions, cfg.hidden)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(obs_dim, cfg.buffer_size, rng)

    def epsilon(self, step):
        c = self.cfg
        frac = min(1.0, step / c.eps_decay_steps)
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def act(self, obs, eps):
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            return int(self.q(torch.from_numpy(obs)[None]).argmax(1).item())

    def update(self, step):
        c = self.cfg
        if step < c.learning_starts or step % c.train_every:
            return
        o, a, r, o2, d = self.buffer.sample(c.batch_size)
        with torch.no_grad():
            a2 = self.q(o2).argmax(1, keepdim=True)
            target = r + c.gamma * (1 - d) * self.q_target(o2).gather(1, a2).squeeze(1)
        q = self.q(o).gather(1, a[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), c.grad_clip)
        self.opt.step()
        if step % c.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())
