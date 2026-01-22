import os
import numpy as np

import torch
import torch.nn as nn

from state_encoder import EncodeState
from networks.multi_critic_sac import MultiCriticSAC
from agents.replay_buffer import ReplayBuffer
from agents.cbf_safety_layer import CBFSafetyLayer
from config import load_config


class MultiSACAgent(object):
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
        self.action_noise_std = cfg['agent'].get('action_noise_std', 0.0)

        self.encode = EncodeState(cfg['env']['latent_dim'])
        self.town = town

        num_critics = cfg['multi_critic']['num_critics']
        ensemble_type = cfg['multi_critic'].get('ensemble_type', 'mean')
        self.model = MultiCriticSAC(
            self.obs_dim, 
            self.action_dim, 
            num_critics=num_critics,
            ensemble_type=ensemble_type
        )
        self.model.to(self.device)
        
        # Trust & Safety parameters
        self.trust_scaling = cfg['trust_safety'].get('trust_scaling', 1.0)
        self.use_trust_aware_loss = cfg['trust_safety'].get('use_trust_aware_loss', False)
        
        # Initialize CBF Safety Layer
        self.safety_layer = CBFSafetyLayer(cfg['trust_safety'])
        self.use_safety_layer = cfg['trust_safety'].get('enable_safety', True)

        self.actor_opt = torch.optim.Adam(self.model.actor.parameters(), lr=self.actor_lr)
        self.q_opts = [torch.optim.Adam(q.parameters(), lr=self.critic_lr) for q in self.model.qs]

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
    
    def save_checkpoint(self, filepath, episode, timestep, avg_reward):
        """
        Save checkpoint including model weights, optimizers, and training state.
        
        Args:
            filepath: Path to save checkpoint
            episode: Current episode number
            timestep: Current timestep
            avg_reward: Average reward for this checkpoint
        """
        checkpoint = {
            'episode': episode,
            'timestep': timestep,
            'avg_reward': avg_reward,
            'model_state_dict': self.model.state_dict(),
            'actor_optimizer': self.actor_opt.state_dict(),
            'critic_optimizers': [opt.state_dict() for opt in self.q_opts],
            'replay_buffer': self.replay,
        }
        
        # Add alpha if auto-tuning
        if self.auto_alpha:
            checkpoint['log_alpha'] = self.log_alpha.item()
            checkpoint['alpha_optimizer'] = self.alpha_opt.state_dict()
        else:
            checkpoint['log_alpha'] = self.log_alpha.item()
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """
        Load checkpoint from file.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Dictionary with episode, timestep, and avg_reward
        """
        if not os.path.exists(filepath):
            print(f"Checkpoint not found at {filepath}")
            return None
            
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.actor_opt.load_state_dict(checkpoint['actor_optimizer'])
        
        for q_opt, state_dict in zip(self.q_opts, checkpoint['critic_optimizers']):
            q_opt.load_state_dict(state_dict)
        
        # Load alpha if auto-tuning
        if self.auto_alpha:
            self.log_alpha.data = torch.tensor(checkpoint['log_alpha'], 
                                                dtype=torch.float32, 
                                                device=self.device, 
                                                requires_grad=True)
            self.alpha_opt.load_state_dict(checkpoint['alpha_optimizer'])
        else:
            self.log_alpha.data = torch.tensor(checkpoint['log_alpha'], 
                                                dtype=torch.float32, 
                                                device=self.device)
        
        self.replay = checkpoint['replay_buffer']
        self.total_steps = checkpoint['timestep']
        
        episode = checkpoint['episode']
        timestep = checkpoint['timestep']
        avg_reward = checkpoint['avg_reward']
        
        print(f"Checkpoint loaded from {filepath}")
        print(f"Resuming from episode {episode}, timestep {timestep}, avg_reward: {avg_reward:.2f}")
        
        return {'episode': episode, 'timestep': timestep, 'avg_reward': avg_reward}
    
    def get_uncertainty(self, obs, action):
        """Estimate epistemic uncertainty using ensemble disagreement."""
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            if isinstance(action, np.ndarray):
                action = torch.tensor(action, dtype=torch.float32, device=self.device)
            uncertainty = self.model.get_uncertainty(obs.unsqueeze(0), action.unsqueeze(0))
            return uncertainty.cpu().numpy()
    
    def get_trust_score(self, obs, action):
        """
        Compute trust score for given state-action pair.
        
        Returns:
            Trust score between 0 and 1 (higher = more trustworthy)
        """
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            if isinstance(action, np.ndarray):
                action = torch.tensor(action, dtype=torch.float32, device=self.device)
            trust_score = self.model.compute_trust_score(obs.unsqueeze(0), action.unsqueeze(0), self.trust_scaling)
            return trust_score.cpu().numpy()

    def select_action(self, obs, evaluate=False, velocity=0.0, return_info=False):
        """
        Select action with optional CBF safety layer filtering
        
        Args:
            obs: Observation
            evaluate: Whether to act deterministically
            velocity: Current vehicle velocity in m/s (for safety layer)
            return_info: Whether to return additional info dict
            
        Returns:
            Action or (action, info) if return_info=True
        """
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            else:
                obs_tensor = obs.to(self.device)
            
            # Get action from actor
            act = self.model.act(obs_tensor.unsqueeze(0), deterministic=evaluate).squeeze(0)
            action = act.cpu().numpy()
            
            # Add exploration noise during training
            if not evaluate and self.action_noise_std > 0:
                noise = np.random.normal(0, self.action_noise_std, size=action.shape)
                action = np.clip(action + noise, -1.0, 1.0)
            
            info = {}
            
            # Apply CBF safety layer if enabled (only during training, not evaluation)
            if self.use_safety_layer and not evaluate:
                # Compute trust score for current action
                trust_score = self.model.compute_trust_score(
                    obs_tensor.unsqueeze(0), 
                    act.unsqueeze(0), 
                    self.trust_scaling
                ).cpu().numpy().item()
                
                # Filter action through safety layer
                obs_array = obs if isinstance(obs, np.ndarray) else obs.cpu().numpy()
                action, safety_info = self.safety_layer.filter_action(
                    action, trust_score, obs_array, velocity
                )
                info.update(safety_info)
            
            if return_info:
                return action, info
            return action

    def push_transition(self, obs, action, reward, next_obs, done):
        # Convert CUDA tensors to CPU numpy arrays if needed
        if torch.is_tensor(obs):
            obs = obs.cpu().numpy()
        if torch.is_tensor(next_obs):
            next_obs = next_obs.cpu().numpy()
        if torch.is_tensor(action):
            action = action.cpu().numpy()
        self.replay.push(obs, action, reward, next_obs, done)
        self.total_steps += 1

    def soft_update_targets(self):
        with torch.no_grad():
            for target_q, q in zip(self.model.target_qs, self.model.qs):
                for target_param, param in zip(target_q.parameters(), q.parameters()):
                    target_param.data.mul_(1 - self.tau)
                    target_param.data.add_(self.tau * param.data)

    def learn(self):
        if self.total_steps < self.warmup_steps or len(self.replay) < self.batch_size:
            # Debug: Print why learning is skipped (only occasionally to avoid spam)
            if self.total_steps % 1000 == 0 and self.total_steps > 0:
                if self.total_steps < self.warmup_steps:
                    print(f"[LEARN] Skipped: warmup ({self.total_steps}/{self.warmup_steps})")
                elif len(self.replay) < self.batch_size:
                    print(f"[LEARN] Skipped: buffer size ({len(self.replay)}/{self.batch_size})")
            return

        for _ in range(self.updates_per_step):
            obs, acts, rews, next_obs, dones = self.replay.sample(self.batch_size)

            obs = obs.to(self.device)
            acts = acts.to(self.device)
            rews = rews.to(self.device)
            next_obs = next_obs.to(self.device)
            dones = dones.to(self.device)

            with torch.no_grad():
                next_action, next_log_prob, _ = self.model.actor.sample(next_obs)
                # Use MINIMUM Q-value from target ensemble to mitigate overestimation bias
                # This is the key for conservative critic training (Trust-SAC-DE)
                target_q_min = self.model.target_q_min(next_obs, next_action)
                target = rews + (1.0 - dones) * self.gamma * (target_q_min - self.alpha * next_log_prob)
                
                # Optional: Compute trust scores for current state-action pairs
                if self.use_trust_aware_loss:
                    trust_scores = self.model.compute_trust_score(obs, acts, self.trust_scaling)

            # Critic updates - each critic trained independently with conservative target
            for idx, (q, opt) in enumerate(zip(self.model.qs, self.q_opts)):
                current_q = q(obs, acts)
                
                if self.use_trust_aware_loss:
                    # Weight loss by trust score: lower trust -> lower weight
                    q_loss = nn.functional.mse_loss(current_q, target, reduction='none')
                    q_loss = (trust_scores * q_loss).mean()
                else:
                    # Standard MSE loss
                    q_loss = nn.functional.mse_loss(current_q, target)
                
                opt.zero_grad()
                q_loss.backward()
                # Clip gradients for training stability
                torch.nn.utils.clip_grad_norm_(q.parameters(), max_norm=1.0)
                opt.step()

            # Policy update using ensemble Q-value
            new_action, log_prob, _ = self.model.actor.sample(obs)
            q_new_ensemble, _, _ = self.model.q_ensemble(obs, new_action)
            actor_loss = (self.alpha * log_prob - q_new_ensemble).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Alpha update
            if self.auto_alpha:
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_opt.zero_grad()
                alpha_loss.backward()
                self.alpha_opt.step()

            self.soft_update_targets()
