"""
Multi-Critic SAC Driver for Autonomous Car System
Main training and testing script for SAC with multi-critic ensemble
"""

import sys
import time
import random
import numpy as np
import argparse
import logging
import traceback
import torch
from distutils.util import strtobool
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from state_encoder import EncodeState
from agents.multi_critic_sac_agent import MultiSACAgent
from simulation.connection import ClientConnection
from simulation.environment import CarlaEnvironment
from config import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Multi-Critic SAC Training for CARLA')
    parser.add_argument('--exp-name', type=str, help='Name of the experiment')
    parser.add_argument('--env-name', type=str, default='carla', help='Name of the simulation environment')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--train', default=True, type=boolean_string, help='Training mode (True) or testing mode (False)')
    parser.add_argument('--town', type=str, default="Town07", help='CARLA town to use')
    parser.add_argument('--load-checkpoint', type=bool, default=False, help='Resume training from checkpoint')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, 
                       nargs='?', const=True, help='Enable deterministic PyTorch operations')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, 
                       nargs='?', const=True, help='Enable CUDA (GPU)')
    return parser.parse_args()


def boolean_string(s):
    """Convert string to boolean."""
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def setup_tensorboard(run_name, total_timesteps, town, is_training):
    """Setup TensorBoard writer."""
    mode = "" if is_training else "_TEST"
    log_dir = f"runs/{run_name}_{int(total_timesteps)}{mode}/{town}"
    writer = SummaryWriter(log_dir)
    return writer


def setup_seeds(cfg, torch_deterministic):
    """Setup random seeds for reproducibility."""
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.backends.cudnn.deterministic = torch_deterministic


def connect_to_carla(town):
    """Connect to CARLA server."""
    try:
        client, world = ClientConnection(town).setup()
        logging.info("Connection has been setup successfully.")
        return client, world
    except Exception as e:
        logging.error(f"Connection has been refused by the server: {e}")
        print(f"Error: Failed to connect to CARLA server. Error: {e}")
        print("Please ensure:")
        print("1. CARLA server is running")
        print(f"2. The town '{town}' exists in your CARLA installation")
        print("3. Client and server versions match")
        sys.exit(1)


def extract_info_metrics(info):
    """Extract metrics from environment info dict."""
    distance_covered = info[0] if len(info) > 0 else 0.0
    deviation_from_center = info[1] if len(info) > 1 else 0.0
    deviation_from_lane = info[2] if len(info) > 2 else 0.0
    deviation_angle = info[3] if len(info) > 3 else 0.0
    collision_occurred = info[4] if len(info) > 4 else False
    return distance_covered, deviation_from_center, deviation_from_lane, deviation_angle, collision_occurred


def log_step_metrics(writer, timestep, deviation_from_lane, deviation_angle, collision_occurred, prefix=""):
    """Log per-step metrics to TensorBoard."""
    writer.add_scalar(f"{prefix}Lane Metrics/Deviation from Lane (m)", deviation_from_lane, timestep)
    writer.add_scalar(f"{prefix}Lane Metrics/Deviation Angle (rad)", deviation_angle, timestep)
    writer.add_scalar(f"{prefix}Lane Metrics/Deviation Angle (deg)", np.degrees(deviation_angle), timestep)
    writer.add_scalar(f"{prefix}Safety/Collision", 1.0 if collision_occurred else 0.0, timestep)


def print_step_info(episode, timestep, reward, done, velocity, deviation_from_lane, 
                    deviation_angle_deg, collision_occurred, prefix=""):
    """Print step information to console."""
    collision_status = "COLLISION!" if collision_occurred else "No collision"
    print(f"{prefix}Episode: {episode}, Timestep: {timestep}, Reward: {reward:.2f}, "
          f"Done: {done}, Velocity: {velocity:.2f} km/h")
    print(f"  -> Deviation from Lane: {deviation_from_lane:.3f} m, "
          f"Deviation Angle: {deviation_angle_deg:.2f}°, Collision: {collision_status}")


def log_episode_metrics(writer, episode, timestep, scores, episodic_length, 
                        deviation_from_center, distance_covered, deviation_from_lane, 
                        deviation_angle, prefix=""):
    """Log episode-level metrics to TensorBoard."""
    # Reward metrics
    writer.add_scalar(f"{prefix}Episodic Reward/episode", scores[-1], episode)
    cumulative_score = np.mean(scores)
    writer.add_scalar(f"{prefix}Cumulative Reward/info", cumulative_score, episode)
    writer.add_scalar(f"{prefix}Cumulative Reward/(t)", cumulative_score, timestep)
    
    # Average reward (last 5 episodes)
    if len(scores) >= 5:
        avg_last_5 = np.mean(scores[-5:])
    else:
        avg_last_5 = np.mean(scores) if len(scores) > 0 else 0.0
    writer.add_scalar(f"{prefix}Average Episodic Reward/info", avg_last_5, episode)
    writer.add_scalar(f"{prefix}Average Reward/(t)", avg_last_5, timestep)
    
    # Episode length
    if episodic_length:
        writer.add_scalar(f"{prefix}Episode Length (s)/info", np.mean(episodic_length), episode)
    
    # Distance and deviation metrics
    episodes_for_avg = 5 if prefix == "" else 1
    writer.add_scalar(f"{prefix}Average Deviation from Center/episode", 
                     deviation_from_center / episodes_for_avg, episode)
    writer.add_scalar(f"{prefix}Average Deviation from Center/(t)", 
                     deviation_from_center / episodes_for_avg, timestep)
    writer.add_scalar(f"{prefix}Average Distance Covered (m)/episode", 
                     distance_covered / episodes_for_avg, episode)
    writer.add_scalar(f"{prefix}Average Distance Covered (m)/(t)", 
                     distance_covered / episodes_for_avg, timestep)
    
    # Lane metrics
    avg_deviation_from_lane = deviation_from_lane / episodes_for_avg
    avg_deviation_angle_rad = deviation_angle / episodes_for_avg
    avg_deviation_angle_deg = np.degrees(avg_deviation_angle_rad)
    
    writer.add_scalar(f"{prefix}Lane Metrics/Average Deviation from Lane (m)/episode", 
                     avg_deviation_from_lane, episode)
    writer.add_scalar(f"{prefix}Lane Metrics/Average Deviation from Lane (m)/(t)", 
                     avg_deviation_from_lane, timestep)
    writer.add_scalar(f"{prefix}Lane Metrics/Average Deviation Angle (rad)/episode", 
                     avg_deviation_angle_rad, episode)
    writer.add_scalar(f"{prefix}Lane Metrics/Average Deviation Angle (rad)/(t)", 
                     avg_deviation_angle_rad, timestep)
    writer.add_scalar(f"{prefix}Lane Metrics/Average Deviation Angle (deg)/episode", 
                     avg_deviation_angle_deg, episode)
    writer.add_scalar(f"{prefix}Lane Metrics/Average Deviation Angle (deg)/(t)", 
                     avg_deviation_angle_deg, timestep)


def train_loop(agent, env, encode, writer, cfg, run_name, town, checkpoint_load, total_timesteps):
    """Main training loop."""
    timestep = 0
    episode = 0
    cumulative_score = 0
    episodic_length = []
    scores = []
    
    # Episode accumulation variables (reset every 5 episodes)
    deviation_from_center = 0
    distance_covered = 0
    deviation_from_lane = 0
    deviation_angle = 0
    
    print(f"Starting training. Target timesteps: {total_timesteps}")
    while timestep < total_timesteps:
        print(f"Resetting environment for new episode. Episode: {episode + 1}")
        observation = env.reset()
        
        # Validate observation format before processing
        if observation is None or len(observation) != 2:
            raise ValueError(f"Invalid observation format. Expected [image, navigation], got: {observation}")
        if observation[0] is None:
            raise ValueError("Image observation is None after reset!")
        
        observation = encode.process(observation)
        
        current_ep_reward = 0
        episode_start_time = datetime.now()
        
        for t in range(cfg['train']['episode_length']):
            # Select action with policy (with safety layer if enabled)
            try:
                action, safety_info = agent.select_action(
                    observation, evaluate=False, velocity=env.velocity, return_info=True
                )
                if timestep % 100 == 0 or timestep < 10:
                    throttle_mapped = (action[1] + 1) / 2
                    print(f"Action: steer={action[0]:.3f}, throttle_raw={action[1]:.3f}, "
                          f"throttle_mapped={throttle_mapped:.3f}, velocity={env.velocity:.2f} km/h")
            except Exception as e:
                print(f"Error in select_action: {e}")
                traceback.print_exc()
                break
            
            # Log safety interventions
            if (cfg['logging'].get('log_safety_interventions', False) and 
                safety_info.get('safety_layer_active', False) and timestep % 100 == 0):
                print(f"[Safety] Trust: {safety_info['trust_score']:.3f}, "
                      f"Modification: {safety_info['action_modification']:.3f}")
            
            # Step environment
            try:
                next_observation, reward, done, info = env.step(action)
            except Exception as e:
                print(f"Error in env.step: {e}")
                traceback.print_exc()
                break
            
            if next_observation is None:
                print("Environment returned None observation. Exiting episode.")
                break
            
            # Process observation
            try:
                next_observation = encode.process(next_observation)
            except Exception as e:
                print(f"Error in encode.process: {e}")
                traceback.print_exc()
                break
            
            # Store transition and learn
            agent.push_transition(observation, action, reward, next_observation, done)
            
            # Debug: Print learning status
            if timestep % 200 == 0:
                if agent.total_steps < agent.warmup_steps:
                    print(f"[WARMUP] Steps: {agent.total_steps}/{agent.warmup_steps}, "
                          f"Replay buffer: {len(agent.replay)}/{agent.batch_size}, "
                          f"Learning: DISABLED")
                else:
                    print(f"[LEARNING] Steps: {agent.total_steps}, "
                          f"Replay buffer: {len(agent.replay)}, Learning: ENABLED")
            
            agent.learn()
            observation = next_observation
            
            timestep += 1
            current_ep_reward += reward
            
            # Extract and log metrics
            distance_covered_step, deviation_from_center_step, deviation_from_lane_step, \
                deviation_angle_step, collision_occurred = extract_info_metrics(info)
            
            deviation_from_lane += abs(deviation_from_lane_step)
            deviation_angle += abs(deviation_angle_step)
            
            log_step_metrics(writer, timestep, deviation_from_lane_step, 
                           deviation_angle_step, collision_occurred)
            
            # Print step info
            if timestep % 100 == 0 or timestep < 10 or collision_occurred:
                print_step_info(episode, timestep, current_ep_reward, done, env.velocity,
                              deviation_from_lane_step, np.degrees(deviation_angle_step),
                              collision_occurred)
            
            # End episode if done
            if done:
                episode += 1
                episode_duration = (datetime.now() - episode_start_time).total_seconds()
                episodic_length.append(abs(episode_duration))
                break
        
        # Update episode-level accumulations
        distance_covered += info[0]
        deviation_from_center += info[1]
        scores.append(current_ep_reward)
        
        # Update cumulative score
        if checkpoint_load:
            cumulative_score = ((cumulative_score * (episode - 1)) + current_ep_reward) / episode
        else:
            cumulative_score = np.mean(scores)
        
        print(f'Episode: {episode}, Timestep: {timestep}, Reward: {current_ep_reward:.2f}, '
              f'Average Reward: {cumulative_score:.2f}')
        
        # Log metrics every 5 episodes
        if episode % 5 == 0:
            log_episode_metrics(writer, episode, timestep, scores, episodic_length,
                              deviation_from_center, distance_covered, deviation_from_lane,
                              deviation_angle)
            
            # Reset accumulators
            episodic_length = []
            deviation_from_center = 0
            distance_covered = 0
            deviation_from_lane = 0
            deviation_angle = 0
        
        # Save checkpoint
        if episode % cfg['train']['save_interval_episodes'] == 0:
            checkpoint_path = f"checkpoints/{run_name}_{town}_episode_{episode}.pt"
            agent.save_checkpoint(checkpoint_path, episode, timestep, cumulative_score)
            
            # Also save as latest checkpoint
            latest_path = f"checkpoints/{run_name}_{town}_latest.pt"
            agent.save_checkpoint(latest_path, episode, timestep, cumulative_score)
    
    print("Training loop completed successfully!")
    print(f"Total episodes: {episode}, Total timesteps: {timestep}")
    
    # Save final checkpoint
    final_checkpoint_path = f"checkpoints/{run_name}_{town}_final.pt"
    agent.save_checkpoint(final_checkpoint_path, episode, timestep, cumulative_score)
    print(f"Final model saved to {final_checkpoint_path}")


def test_loop(agent, env, encode, writer, cfg):
    """Main testing loop."""
    timestep = 0
    episode = 0
    episodic_length = []
    scores = []
    
    # Episode accumulation variables
    deviation_from_center = 0
    distance_covered = 0
    deviation_from_lane = 0
    deviation_angle = 0
    
    while timestep < cfg['train']['test_timesteps']:
        observation = env.reset()
        observation = encode.process(observation)
        
        current_ep_reward = 0
        episode_start_time = datetime.now()
        
        for t in range(cfg['train']['episode_length']):
            # Select action with policy (deterministic for testing)
            action, _ = agent.select_action(observation, evaluate=True, return_info=True)
            observation, reward, done, info = env.step(action)
            
            if observation is None:
                break
            
            observation = encode.process(observation)
            timestep += 1
            current_ep_reward += reward
            
            # Extract and log metrics
            distance_covered_step, deviation_from_center_step, deviation_from_lane_step, \
                deviation_angle_step, collision_occurred = extract_info_metrics(info)
            
            deviation_from_lane += abs(deviation_from_lane_step)
            deviation_angle += abs(deviation_angle_step)
            
            log_step_metrics(writer, timestep, deviation_from_lane_step,
                           deviation_angle_step, collision_occurred, prefix="TEST: ")
            
            # Print step info
            if timestep % 100 == 0 or timestep < 10 or collision_occurred:
                print_step_info(episode, timestep, current_ep_reward, done, env.velocity,
                              deviation_from_lane_step, np.degrees(deviation_angle_step),
                              collision_occurred, prefix="TEST - ")
            
            # End episode if done
            if done:
                episode += 1
                episode_duration = (datetime.now() - episode_start_time).total_seconds()
                episodic_length.append(abs(episode_duration))
                break
        
        # Update episode-level accumulations
        distance_covered += info[0]
        deviation_from_center += info[1]
        scores.append(current_ep_reward)
        cumulative_score = np.mean(scores)
        
        print(f'Episode: {episode}, Timestep: {timestep}, Reward: {current_ep_reward:.2f}, '
              f'Average Reward: {cumulative_score:.2f}')
        
        # Log metrics for each episode
        log_episode_metrics(writer, episode, timestep, scores, episodic_length,
                          deviation_from_center, distance_covered, deviation_from_lane,
                          deviation_angle, prefix="TEST: ")
        
        # Reset accumulators
        episodic_length = []
        deviation_from_center = 0
        distance_covered = 0
        deviation_from_lane = 0
        deviation_angle = 0
    
    print("Testing complete. Terminating the run.")


def main():
    """Main function."""
    # Parse arguments and load config
    args = parse_args()
    cfg = load_config(args.config)
    
    # Setup
    run_name = "SAC_MultiCritic"
    total_timesteps = cfg['train']['total_timesteps']
    image_capture_cfg = cfg.get('image_capture', {})
    
    # Setup TensorBoard
    writer = setup_tensorboard(run_name, total_timesteps, args.town, args.train)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in vars(args).items()]))
    )
    
    # Setup seeds
    setup_seeds(cfg, args.torch_deterministic)
    
    # Connect to CARLA
    client, world = connect_to_carla(args.town)
    
    # Create environment and encoder
    if args.train:
        env = CarlaEnvironment(
            client,
            world,
            args.town,
            capture_config=image_capture_cfg,
        )
    else:
        env = CarlaEnvironment(
            client,
            world,
            args.town,
            checkpoint_frequency=None,
            capture_config=image_capture_cfg,
        )
    encode = EncodeState(cfg['env']['latent_dim'])
    
    # Create agent
    try:
        time.sleep(0.5)  # Brief pause for initialization
        agent = MultiSACAgent(args.town, args.config)
        
        # Load checkpoint if requested
        if args.load_checkpoint and args.train:
            checkpoint_path = f"checkpoints/{run_name}_{args.town}_latest.pt"
            checkpoint_info = agent.load_checkpoint(checkpoint_path)
            if checkpoint_info:
                timestep = checkpoint_info['timestep']
                episode = checkpoint_info['episode']
                print(f"Loaded checkpoint: episode={episode}, timestep={timestep}")
        
        # Run training or testing
        if args.train:
            train_loop(agent, env, encode, writer, cfg, run_name, args.town, 
                      args.load_checkpoint, total_timesteps)
        else:
            test_loop(agent, env, encode, writer, cfg)
    
    finally:
        print("Cleaning up...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print('\nExit')
