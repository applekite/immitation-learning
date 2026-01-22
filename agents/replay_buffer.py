import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, n_step: int = 1, gamma: float = 0.99, device: str = 'cpu'):
        self.capacity = capacity
        self.n_step = max(1, n_step)
        self.gamma = gamma
        self.device = torch.device(device)

        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rews_buf = np.zeros((capacity,), dtype=np.float32)
        self.dones_buf = np.zeros((capacity,), dtype=np.float32)

        self.nstep_queue = deque(maxlen=self.n_step)
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def _get_nstep_info(self):
        reward, next_obs, done = 0.0, None, 0.0
        for idx, (obs, act, rew, nobs, dn) in enumerate(self.nstep_queue):
            reward += (self.gamma ** idx) * rew
            next_obs = nobs
            done = max(done, float(dn))
            if dn:
                break
        return reward, next_obs, done

    def push(self, obs, action, reward, next_obs, done):
        self.nstep_queue.append((obs, action, reward, next_obs, done))
        if len(self.nstep_queue) < self.n_step:
            return

        R, nobs, dn = self._get_nstep_info()
        o, a, _, _, _ = self.nstep_queue[0]

        self.obs_buf[self.ptr] = o
        self.acts_buf[self.ptr] = a
        self.rews_buf[self.ptr] = R
        self.next_obs_buf[self.ptr] = nobs
        self.dones_buf[self.ptr] = dn

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs = torch.as_tensor(self.obs_buf[idxs], device=self.device)
        acts = torch.as_tensor(self.acts_buf[idxs], device=self.device)
        rews = torch.as_tensor(self.rews_buf[idxs], device=self.device).unsqueeze(-1)
        next_obs = torch.as_tensor(self.next_obs_buf[idxs], device=self.device)
        dones = torch.as_tensor(self.dones_buf[idxs], device=self.device).unsqueeze(-1)
        return obs, acts, rews, next_obs, dones


