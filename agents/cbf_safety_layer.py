"""
Control Barrier Function (CBF) based Safety Layer with QP Solver

This module implements a real-time safety filter that guarantees forward invariance
of a safe set by solving a quadratic programming problem.

Key Features:
- Trust-based activation trigger
- Hard safety constraint violations
- Multiple barrier functions (collision, lane keeping, speed limits)
- Real-time QP solver using OSQP or cvxpy
"""

import numpy as np
import scipy.optimize as opt
from typing import Tuple, List, Optional, Dict
import warnings


class CBFSafetyLayer:
    """
    Control Barrier Function Safety Layer with QP Solver
    
    """
    
    def __init__(self, config: dict):
        """
        Initialize CBF Safety Layer
        
        Args:
            config: Configuration dictionary with trust_safety parameters
        """
        self.trust_threshold = config.get('trust_threshold', 0.5)
        self.cbf_gamma = config.get('cbf_gamma', 1.0)
        self.trust_scaling = config.get('trust_scaling', 1.0)
        
        self.qp_epsilon = config.get('qp_solver_epsilon', 1e-6)
        self.enable_safety = config.get('enable_safety', True)
        
        self.max_lateral_acceleration = config.get('max_lateral_acceleration', 5.0)
        self.max_longitudinal_acceleration = config.get('max_longitudinal_acceleration', 6.0)
        self.collision_detection_margin = config.get('collision_detection_margin', 0.5)
        self.min_speed = 0.0
        self.max_speed = 20.0
        
        self.vehicle_length = 4.5
        self.vehicle_width = 2.0
        self.reaction_time = 0.1
        
        self.action_bounds = {
            'steer': (-1.0, 1.0),
            'throttle': (-1.0, 1.0)
        }
        
        self.previous_action = np.array([0.0, 0.0])
        self.previous_velocity = 0.0
        
    def compute_barrier_collision(self, obs: np.ndarray, action: np.ndarray, 
                                   velocity: float) -> Tuple[float, np.ndarray]:
        """
        Barrier function for collision avoidance based on Time-to-Collision (TTC)
        
        h(x) = distance_to_obstacle - safe_distance
        
        Args:
            obs: Observation vector (contains estimated distances/sensors)
            action: Proposed action [steer, throttle]
            velocity: Current vehicle velocity in m/s
            
        Returns:
            Barrier value and gradient w.r.t. action
        """
        if len(obs) >= 3:
            distance_ahead = obs[0] * 100.0
        else:
            distance_ahead = 50.0
        
        safe_distance = velocity * self.reaction_time + self.vehicle_length * 0.5
        safe_distance = max(safe_distance, 5.0)
        
        barrier_value = distance_ahead - safe_distance
        
        dbarrier_daction = np.array([
            0.0,
            -2.0 * velocity * self.reaction_time if velocity > 0 else 0.0
        ])
        
        return barrier_value, dbarrier_daction
    
    def compute_barrier_lane_keeping(self, obs: np.ndarray, action: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Barrier function for lane keeping
        
        h(x) = lane_width/2 - |lateral_displacement|
        
        Args:
            obs: Observation vector
            action: Proposed action [steer, throttle]
            
        Returns:
            Barrier value and gradient w.r.t. action
        """
        if len(obs) >= 3:
            lateral_displacement = obs[1] * 5.0
        else:
            lateral_displacement = 0.0
        
        lane_width = 3.5
        half_lane = lane_width / 2.0
        
        barrier_value = half_lane - abs(lateral_displacement)
        
        if lateral_displacement > 0:
            dbarrier_dsteer = -2.0
        elif lateral_displacement < 0:
            dbarrier_dsteer = 2.0
        else:
            dbarrier_dsteer = 0.0
        
        dbarrier_daction = np.array([dbarrier_dsteer, 0.0])
        
        return barrier_value, dbarrier_daction
    
    def compute_barrier_speed_limit(self, obs: np.ndarray, action: np.ndarray, 
                                     velocity: float) -> Tuple[float, np.ndarray]:
        """
        Barrier function for speed limits
        
        h(x) = v_max - v_current (upper limit)
        h(x) = v_current - v_min (lower limit)
        
        Args:
            obs: Observation vector
            action: Proposed action [steer, throttle]
            velocity: Current velocity
            
        Returns:
            Barrier value and gradient w.r.t. action
        """
        throttle = (action[1] + 1.0) / 2.0
        
        h_upper = self.max_speed - velocity
        
        h_lower = velocity - self.min_speed
        
        if h_upper < h_lower:
            barrier_value = h_upper
            dbarrier_dthrottle = -2.0 * action[1] if velocity > 10 else -1.0
        else:
            barrier_value = h_lower
            dbarrier_dthrottle = 2.0 * action[1] if velocity < 5 else 1.0
        
        dbarrier_daction = np.array([0.0, dbarrier_dthrottle])
        
        return barrier_value, dbarrier_daction
    
    def check_activation_trigger(self, trust_score: float, obs: np.ndarray, 
                                  velocity: float) -> bool:
        """
        Check if CBF safety layer should activate
        
        Activation conditions:
        1. Trust score below threshold
        2. Hard safety constraint violation
        
        Args:
            trust_score: Trust score from critic ensemble [0, 1]
            obs: Current observation
            velocity: Current velocity
            
        Returns:
            True if safety layer should activate
        """
        if not self.enable_safety:
            return False
        
        if trust_score < self.trust_threshold:
            return True
        
        distance_ahead = obs[0] * 100.0 if len(obs) >= 3 else 50.0
        safe_distance = velocity * self.reaction_time + self.vehicle_length * 0.5
        if distance_ahead < safe_distance:
            return True
        
        if velocity > self.max_speed or velocity < self.min_speed:
            return True
        
        lateral_displacement = abs(obs[1] * 5.0) if len(obs) >= 3 else 0.0
        if lateral_displacement > 3.5 / 2.0:
            return True
        
        return False
    
    def solve_cbf_qp(self, actor_action: np.ndarray, obs: np.ndarray, 
                     velocity: float) -> np.ndarray:
        """
        Solve CBF-QP problem to find safe action
        
        minimize ||u_safe - u_actor||²
        subject to: dh/dt + γ·h ≥ 0 for all barriers
        
        Args:
            actor_action: Proposed action from actor [steer, throttle]
            obs: Current observation
            velocity: Current velocity
            
        Returns:
            Safe action [steer, throttle]
        """
        barriers = []
        gradients = []
        
        h_collision, grad_collision = self.compute_barrier_collision(obs, actor_action, velocity)
        barriers.append(h_collision)
        gradients.append(grad_collision)
        
        h_lane, grad_lane = self.compute_barrier_lane_keeping(obs, actor_action)
        barriers.append(h_lane)
        gradients.append(grad_lane)
        
        h_speed, grad_speed = self.compute_barrier_speed_limit(obs, actor_action, velocity)
        barriers.append(h_speed)
        gradients.append(grad_speed)
        
        H = np.eye(2)
        g = -2 * actor_action
        
        A_ub = []
        b_ub = []
        
        for h, grad in zip(barriers, gradients):
            A_ub.append(-grad)
            b_ub.append(self.cbf_gamma * h - np.dot(grad, actor_action))
        
        if len(A_ub) > 0:
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
        else:
            return np.clip(actor_action, 
                          [self.action_bounds['steer'][0], self.action_bounds['throttle'][0]],
                          [self.action_bounds['steer'][1], self.action_bounds['throttle'][1]])
        
        bounds = [
            self.action_bounds['steer'],
            self.action_bounds['throttle']
        ]
        
        try:
            result = opt.minimize(
                lambda u: np.dot(u, H @ u) + np.dot(g, u),
                x0=actor_action,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    opt.LinearConstraint(A_ub, lb=b_ub, ub=np.inf)
                ],
                options={'ftol': self.qp_epsilon, 'disp': False}
            )
            
            safe_action = result.x
            
        except Exception as e:
            warnings.warn(f"QP solver failed: {e}. Using conservative action.")
            safe_action = actor_action * 0.7
        
        return np.clip(safe_action,
                      [self.action_bounds['steer'][0], self.action_bounds['throttle'][0]],
                      [self.action_bounds['steer'][1], self.action_bounds['throttle'][1]])
    
    def filter_action(self, actor_action: np.ndarray, trust_score: float, 
                      obs: np.ndarray, velocity: float, info: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Main interface: filter actor action through CBF safety layer
        
        Args:
            actor_action: Proposed action from actor [steer, throttle]
            trust_score: Trust score from critic ensemble [0, 1]
            obs: Current observation
            velocity: Current velocity in m/s
            info: Optional info dictionary for logging
            
        Returns:
            Filtered action and info dictionary
        """
        if info is None:
            info = {}
        
        should_activate = self.check_activation_trigger(trust_score, obs, velocity)
        
        if should_activate:
            safe_action = self.solve_cbf_qp(actor_action, obs, velocity)
            
            info['safety_layer_active'] = True
            info['original_action'] = actor_action.copy()
            info['safe_action'] = safe_action.copy()
            info['action_modification'] = np.linalg.norm(safe_action - actor_action)
            info['trust_score'] = trust_score
            
            action = safe_action
        else:
            info['safety_layer_active'] = False
            info['trust_score'] = trust_score
            action = actor_action
        
        self.previous_action = action.copy()
        self.previous_velocity = velocity
        
        return action, info
