import math
import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims, output_dim: int, output_activation=None):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.output_activation = output_activation

    def forward(self, x):
        x = self.net(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_dims, 2 * action_dim)
        self.action_dim = action_dim

    def forward(self, obs):
        mean_logstd = self.backbone(obs)
        mean, log_std = torch.chunk(mean_logstd, 2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, std, log_std

    def sample(self, obs):
        mean, std, log_std = self.forward(obs)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        # Correction for Tanh squashing
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        self.q = MLP(obs_dim + action_dim, hidden_dims, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q(x)


class TwinCriticSAC(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        self.actor = GaussianPolicy(obs_dim, action_dim, hidden_dims)
        self.q1 = QNetwork(obs_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_dims)
        # target critics
        self.target_q1 = QNetwork(obs_dim, action_dim, hidden_dims)
        self.target_q2 = QNetwork(obs_dim, action_dim, hidden_dims)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False):
        if deterministic:
            mean, _, _ = self.actor.forward(obs)
            action = torch.tanh(mean)
            return action
        action, _, _ = self.actor.sample(obs)
        return action
