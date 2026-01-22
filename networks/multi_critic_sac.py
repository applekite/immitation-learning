import torch
import torch.nn as nn
import numpy as np

from .twin_critic_sac import GaussianPolicy, QNetwork


class MultiCriticSAC(nn.Module):
    """
    Deep Ensemble SAC with multiple critics.
    
    Deep ensemble properties:
    - Multiple independently initialized critic networks
    - Each critic is trained independently
    - Ensemble statistics (mean, std, min) used for value estimation
    """
    def __init__(self, obs_dim: int, action_dim: int, num_critics: int = 5, 
                 hidden_dims=(256, 256), ensemble_type='mean'):
        super().__init__()
        assert num_critics >= 2, "Use at least 2 critics for ensemble"
        assert ensemble_type in ['mean', 'min', 'weighted_mean'], "ensemble_type must be 'mean', 'min', or 'weighted_mean'"
        
        self.num_critics = num_critics
        self.ensemble_type = ensemble_type
        
        # Actor network
        self.actor = GaussianPolicy(obs_dim, action_dim, hidden_dims)
        
        # Deep ensemble of critics with diverse initialization
        self.qs = nn.ModuleList()
        for i in range(num_critics):
            # Initialize each critic with different random seeds for diversity
            critic = QNetwork(obs_dim, action_dim, hidden_dims)
            # Apply different initialization to encourage diversity
            self._init_critic_with_diversity(critic, seed=i)
            self.qs.append(critic)
        
        # Target critics
        self.target_qs = nn.ModuleList([QNetwork(obs_dim, action_dim, hidden_dims) for _ in range(num_critics)])
        for i in range(num_critics):
            self._init_critic_with_diversity(self.target_qs[i], seed=i)
            # Load initial state from main critics
            self.target_qs[i].load_state_dict(self.qs[i].state_dict())

    def _init_critic_with_diversity(self, network, seed=None):
        """Initialize network with diverse weights to encourage ensemble diversity."""
        if seed is not None:
            rng = np.random.RandomState(seed)
            torch_seed = rng.randint(0, 2**31)
            torch.manual_seed(torch_seed)
        
        # Apply different initialization scales to different critics
        for param in network.parameters():
            if len(param.shape) >= 2:  # Weight matrices
                # Use Xavier/Glorot initialization with some variation
                nn.init.xavier_uniform_(param, gain=1.0 + seed * 0.1 if seed else 1.0)
            else:  # Bias vectors
                nn.init.constant_(param, 0.0)

    def q_ensemble(self, obs, action):
        """
        Compute ensemble Q-value predictions.
        
        Args:
            obs: Observation tensor
            action: Action tensor
            ensemble_type: 'mean', 'min', or 'weighted_mean'
            
        Returns:
            Mean Q-value, std Q-value, individual Q-values
        """
        q_values = [q(obs, action) for q in self.qs]
        q_values = torch.stack(q_values, dim=0)  # Shape: [num_critics, batch_size, 1]
        
        mean_q = torch.mean(q_values, dim=0)
        std_q = torch.std(q_values, dim=0)
        
        if self.ensemble_type == 'mean':
            ensemble_q = mean_q
        elif self.ensemble_type == 'min':
            ensemble_q = torch.min(q_values, dim=0)[0]
        elif self.ensemble_type == 'weighted_mean':
            # Weight by inverse uncertainty (lower std -> higher weight)
            weights = 1.0 / (std_q + 1e-6)
            weights = weights / torch.sum(weights, dim=0, keepdim=True)
            ensemble_q = torch.sum(weights * q_values, dim=0)
        else:
            ensemble_q = mean_q
        
        return ensemble_q, std_q, q_values
    
    def target_q_ensemble(self, obs, action):
        """
        Compute ensemble Q-value predictions from target networks.
        
        Args:
            obs: Observation tensor
            action: Action tensor
            
        Returns:
            Ensemble Q-value prediction based on ensemble_type
        """
        q_values = [tq(obs, action) for tq in self.target_qs]
        q_values = torch.stack(q_values, dim=0)  # Shape: [num_critics, batch_size, 1]
        
        if self.ensemble_type == 'mean':
            return torch.mean(q_values, dim=0)
        elif self.ensemble_type == 'min':
            return torch.min(q_values, dim=0)[0]
        elif self.ensemble_type == 'weighted_mean':
            std_q = torch.std(q_values, dim=0)
            weights = 1.0 / (std_q + 1e-6)
            weights = weights / torch.sum(weights, dim=0, keepdim=True)
            return torch.sum(weights * q_values, dim=0)
        else:
            return torch.mean(q_values, dim=0)
    
    def target_q_min(self, obs, action):
        """
        Get minimum Q-value from target ensemble (conservative approach).
        
        This mitigates overestimation bias by using the most conservative 
        Q-value estimate from the ensemble, similar to clipped double Q-learning 
        but extended to multiple critics.
        
        Args:
            obs: Observation tensor
            action: Action tensor
            
        Returns:
            Minimum Q-value across all target critics
        """
        q_values = [tq(obs, action) for tq in self.target_qs]
        q_values = torch.stack(q_values, dim=0)  # Shape: [num_critics, batch_size, 1]
        return torch.min(q_values, dim=0)[0]  # Return minimum across critics
    
    def get_uncertainty(self, obs, action):
        """
        Estimate epistemic uncertainty from ensemble disagreement.
        
        Args:
            obs: Observation tensor
            action: Action tensor
            
        Returns:
            Standard deviation across ensemble (uncertainty measure)
        """
        q_values = [q(obs, action) for q in self.qs]
        q_values = torch.stack(q_values, dim=0)
        return torch.std(q_values, dim=0)
    
    def get_ensemble_variance(self, obs, action):
        """
        Calculate variance of Q-value predictions across ensemble.
        
        Args:
            obs: Observation tensor
            action: Action tensor
            
        Returns:
            Variance of Q-values across critics
        """
        q_values = [q(obs, action) for q in self.qs]
        q_values = torch.stack(q_values, dim=0)  # Shape: [num_critics, batch_size, 1]
        variance = torch.var(q_values, dim=0)
        return variance
    
    def compute_trust_score(self, obs, action, trust_scaling=1.0):
        """
        Compute trust score from ensemble variance.
        
        Trust Calculation:
        1. Get K Q-value predictions: Q_ensemble(s, a) = {Q1, Q2, ..., QK}
        2. Calculate variance: Var(Q_ensemble)
        3. Convert to trust score: C(s, a) = exp(-λ ⋅ Var(Q_ensemble))
        
        High variance → Low trust → Novel/out-of-distribution state
        
        Args:
            obs: Observation tensor
            action: Action tensor
            trust_scaling: Lambda (λ) - tunable scaling hyperparameter
            
        Returns:
            Trust score between 0 and 1
        """
        variance = self.get_ensemble_variance(obs, action)
        trust_score = torch.exp(-trust_scaling * variance)
        return trust_score

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False):
        if deterministic:
            mean, _, _ = self.actor.forward(obs)
            action = torch.tanh(mean)
            return action
        action, _, _ = self.actor.sample(obs)
        return action