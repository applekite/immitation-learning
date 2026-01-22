import os
import numpy as np

import torch
import torch.nn as nn

from state_encoder import EncodeState
from networks.twin_critic_sac import TwinCriticSAC
from agents.replay_buffer import ReplayBuffer
from config import load_config


class TwinSACAgent(object):
    def __init__(self, town, config_path=None):
        cfg = load_config(config_path)
        device = cfg.get('device', 'cpu')
        self.device = torch.device(device)

        self.obs_dim = cfg['agent']['obs_dim']
        self.action_dim = cfg['agent']['action_dim']
        self.gamma = cfg['sac']['gamma']
        self.tau = cfg['sac']['tau']
        self.auto_alpha = cfg['sac']['auto_alpha']
        self.target_entropy = -cfg['agent']['action_dim'] * cfg['sac']['target_entropy_coeff']

        self.actor_lr = cfg['sac']['actor_lr']
        self.critic_lr = cfg['sac']['critic_lr']
        self.alpha_lr = cfg['sac']['alpha_lr']

        self.batch_size = cfg['train']['batch_size']
        self.gradient_steps = cfg['train']['gradient_steps']
        self.updates_per_step = cfg['train']['updates_per_step']
        self.warmup_steps = cfg['train']['warmup_steps']

        self.encode = EncodeState(cfg['env']['latent_dim'])
        self.town = town

        self.model = TwinCriticSAC(self.obs_dim, self.action_dim)
        self.model.to(self.device)

        self.actor_opt = torch.optim.Adam(self.model.actor.parameters(), lr=self.actor_lr)
        self.q1_opt = torch.optim.Adam(self.model.q1.parameters(), lr=self.critic_lr)
        self.q2_opt = torch.optim.Adam(self.model.q2.parameters(), lr=self.critic_lr)

        if self.auto_alpha:
            self.log_alpha = torch.tensor(np.log(cfg['sac']['init_alpha']), dtype=torch.float32, device=self.device, requires_grad=True)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.log_alpha = torch.tensor(np.log(cfg['sac']['init_alpha']), dtype=torch.float32, device=self.device)

        self.replay = ReplayBuffer(
            capacity=cfg['replay_buffer']['capacity'],
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            n_step=cfg['replay_buffer']['n_step'],
            gamma=self.gamma,
            device=device,
        )

        self.total_steps = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs, evaluate=False):
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            act = self.model.act(obs.unsqueeze(0), deterministic=evaluate).squeeze(0)
            return act.cpu().numpy()

    def push_transition(self, obs, action, reward, next_obs, done):
        self.replay.push(obs, action, reward, next_obs, done)
        self.total_steps += 1

    def soft_update_targets(self):
        with torch.no_grad():
            for target_param, param in zip(self.model.target_q1.parameters(), self.model.q1.parameters()):
                target_param.data.mul_(1 - self.tau)
                target_param.data.add_(self.tau * param.data)
            for target_param, param in zip(self.model.target_q2.parameters(), self.model.q2.parameters()):
                target_param.data.mul_(1 - self.tau)
                target_param.data.add_(self.tau * param.data)

    def learn(self):
        if self.total_steps < self.warmup_steps or len(self.replay) < self.batch_size:
            return

        for _ in range(self.updates_per_step):
            obs, acts, rews, next_obs, dones = self.replay.sample(self.batch_size)

            # Move to device
            obs = obs.to(self.device)
            acts = acts.to(self.device)
            rews = rews.to(self.device)
            next_obs = next_obs.to(self.device)
            dones = dones.to(self.device)

            with torch.no_grad():
                next_action, next_log_prob, _ = self.model.actor.sample(next_obs)
                target_q1 = self.model.target_q1(next_obs, next_action)
                target_q2 = self.model.target_q2(next_obs, next_action)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
                target = rews + (1.0 - dones) * self.gamma * target_q

            current_q1 = self.model.q1(obs, acts)
            current_q2 = self.model.q2(obs, acts)
            q1_loss = nn.functional.mse_loss(current_q1, target)
            q2_loss = nn.functional.mse_loss(current_q2, target)

            self.q1_opt.zero_grad()
            q1_loss.backward()
            self.q1_opt.step()

            self.q2_opt.zero_grad()
            q2_loss.backward()
            self.q2_opt.step()

            # Policy update
            new_action, log_prob, _ = self.model.actor.sample(obs)
            q1_new = self.model.q1(obs, new_action)
            q2_new = self.model.q2(obs, new_action)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (self.alpha * log_prob - q_new).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Alpha update
            if self.auto_alpha:
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_opt.zero_grad()
                alpha_loss.backward()
                self.alpha_opt.step()

            # target networks
            self.soft_update_targets()
