import time
import random
import numpy as np
import pygame
from simulation.connection import carla
from simulation.sensors import CameraSensor, CameraSensorEnv, CollisionSensor
from simulation.settings import *
from manual_control.capture_image import SemanticImageRecorder


class CarlaEnvironment():

    def __init__(
        self,
        client,
        world,
        town,
        checkpoint_frequency=100,
        continuous_action=True,
        enable_auto_correction=False,
        capture_config=None,
    ) -> None:


        self.client = client
        self.enable_auto_correction = enable_auto_correction  # Disable by default to let agent learn
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.action_space = self.get_discrete_action_space()
        self.continous_action_space = continuous_action
        self.display_on = VISUAL_DISPLAY
        self.vehicle = None
        self.settings = None
        self.current_waypoint_index = 0
        self.checkpoint_waypoint_index = 0
        self.fresh_start=True
        self.checkpoint_frequency = checkpoint_frequency
        self.route_waypoints = None
        self.town = town
        self.capture_config = capture_config or {}
        self.capture_enabled = bool(self.capture_config.get("enabled", False))
        self.image_recorder = None
        
        # Objects to be kept alive
        self.camera_obj = None
        self.env_camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None

        # Two very important lists for keeping track of our actors and their observations.
        self.sensor_list = list()
        self.actor_list = list()
        self.walker_list = list()
        self.create_pedestrians()
        if self.capture_enabled:
            self._initialize_image_recorder()

    def _initialize_image_recorder(self):
        try:
            output_dir = self.capture_config.get("output_dir", "captures/agent")
            interval_seconds = self.capture_config.get("interval_ms", 125) / 1000.0
            width = int(self.capture_config.get("width", 256))
            height = int(self.capture_config.get("height", 128))
            prefix = self.capture_config.get("prefix", "agent_semantic")

            self.image_recorder = SemanticImageRecorder(
                output_dir=output_dir,
                capture_interval=interval_seconds,
                width=width,
                height=height,
                prefix=prefix,
            )
            print(f"[ImageCapture] Enabled. Saving semantic frames to '{output_dir}'")
        except Exception as e:
            print(f"[ImageCapture] Failed to initialize recorder: {e}")
            self.image_recorder = None
            self.capture_enabled = False



    # A reset function for reseting our environment.
    def reset(self):

        try:
            
            if len(self.actor_list) != 0 or len(self.sensor_list) != 0:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
                self.sensor_list.clear()
                self.actor_list.clear()
            self.remove_sensors()


            # Blueprint of our main vehicle
            vehicle_bp = self.get_vehicle(CAR_NAME)

            if self.town == "Town07":
                transform = self.map.get_spawn_points()[38] #Town7  is 38 
                self.total_distance = 750
            elif self.town == "Town02":
                transform = self.map.get_spawn_points()[1] #Town2 is 1
                self.total_distance = 780
            else:
                transform = random.choice(self.map.get_spawn_points())
                self.total_distance = 250

            self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
            self.actor_list.append(self.vehicle)
            
            # Apply initial velocity boost to get the car moving at high speed
            # 12 m/s = ~43 km/h (good starting speed for training)
            initial_velocity = carla.Vector3D(x=12.0, y=0.0, z=0.0)  # 43 km/h forward
            self.vehicle.set_target_velocity(initial_velocity)
            
            # Also apply initial throttle control to maintain speed
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.1, steer=0.0))


            # Camera Sensor
            self.camera_obj = CameraSensor(self.vehicle)
            while(len(self.camera_obj.front_camera) == 0):
                time.sleep(0.0001)
            
            self.sensor_list.append(self.camera_obj.sensor)

            # Third person view of our vehicle in the Simulated env
            if self.display_on:
                self.env_camera_obj = CameraSensorEnv(self.vehicle)
                self.sensor_list.append(self.env_camera_obj.sensor)

            # Collision sensor
            self.collision_obj = CollisionSensor(self.vehicle)
            self.collision_history = self.collision_obj.collision_data
            self.sensor_list.append(self.collision_obj.sensor)

            
            self.timesteps = 0
            self.rotation = self.vehicle.get_transform().rotation.yaw
            self.previous_location = self.vehicle.get_location()
            self.distance_traveled = 0.0
            self.center_lane_deviation = 0.0
            self.target_speed = 40 #km/h (increased from 22)
            self.max_speed = 60.0  #km/h (increased from 25)
            self.min_speed = 20.0  #km/h (increased from 15)
            self.max_distance_from_center = 5  # Increased from 3 to 5 meters for more lenient reset condition
            self.throttle = float(0.8)  # Start with high throttle to maintain speed
            self.previous_steer = float(0.0)  # Reset steering to prevent bias accumulation
            self.velocity = float(43.2)  # Initialize to match initial velocity (12 m/s = 43.2 km/h)
            self.distance_from_center = float(0.0)
            self.angle = float(0.0)
            self.center_lane_deviation = 0.0
            self.distance_covered = 0.0
            self.previous_waypoint_index = 0  # Initialize for progress tracking


            if self.fresh_start:
                self.current_waypoint_index = 0
                # Waypoint nearby angle and distance from it
                self.route_waypoints = list()
                self.waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
                current_waypoint = self.waypoint
                self.route_waypoints.append(current_waypoint)
                for x in range(self.total_distance):
                    if self.town == "Town07":
                        if x < 650:
                            next_waypoint = current_waypoint.next(1.0)[0]
                        else:
                            next_waypoint = current_waypoint.next(1.0)[-1]
                    elif self.town == "Town02":
                        if x < 650:
                            next_waypoint = current_waypoint.next(1.0)[-1]
                        else:
                            next_waypoint = current_waypoint.next(1.0)[0]
                    else:
                        next_waypoint = current_waypoint.next(1.0)[0]
                    self.route_waypoints.append(next_waypoint)
                    current_waypoint = next_waypoint
            else:
                # Teleport vehicle to last checkpoint
                waypoint = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)]
                transform = waypoint.transform
                self.vehicle.set_transform(transform)
                self.current_waypoint_index = self.checkpoint_waypoint_index

            time.sleep(0.5)
            self.collision_history.clear()
            # Initialize previous collision count for tracking new collisions
            if self.collision_obj is not None:
                self.previous_collision_count = len(self.collision_obj.collision_data)
            else:
                self.previous_collision_count = 0
            # Initialize previous collision count for tracking new collisions
            if self.collision_obj is not None:
                self.previous_collision_count = len(self.collision_obj.collision_data)
            else:
                self.previous_collision_count = 0

            self.episode_start_time = time.time()
            # Get observation using the observation function
            return self._get_observation()

        except:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()


# ----------------------------------------------------------------
# Step method is used for implementing actions taken by our agent|
# ----------------------------------------------------------------

    def _get_observation(self):
        """
        Extract observation (image and navigation data) from the environment.
        This function is called both during reset() and step() to get current observations.
        
        Returns:
            List containing [image_obs, navigation_obs]
        """
        # Check if camera is initialized
        if self.camera_obj is None:
            raise RuntimeError("Camera object is not initialized! Call reset() first to initialize the environment.")
        
        # Wait for camera image to be available
        while(len(self.camera_obj.front_camera) == 0):
            time.sleep(0.0001)

        # Get image from camera - ensure it's a valid numpy array
        if len(self.camera_obj.front_camera) == 0:
            raise RuntimeError("Camera has no images available!")
        self.image_obs = self.camera_obj.front_camera.pop(-1)
        
        # Validate image observation format
        if self.image_obs is None:
            raise RuntimeError("Camera returned None image!")
        if not isinstance(self.image_obs, np.ndarray):
            raise RuntimeError(f"Camera image is not a numpy array! Type: {type(self.image_obs)}, Value: {self.image_obs}")
        if self.image_obs.ndim < 2:
            raise RuntimeError(f"Camera image has wrong dimensions! Shape: {self.image_obs.shape}, Expected at least 2D (H, W) or 3D (H, W, C)")
        if self.image_obs.size == 0:
            raise RuntimeError(f"Camera image is empty! Shape: {self.image_obs.shape}")
        if self.image_recorder is not None:
            try:
                self.image_recorder.save(self.image_obs)
            except Exception as capture_error:
                print(f"[ImageCapture] Failed to save frame: {capture_error}")
                self.image_recorder = None
        
        # Compute navigation observation
        # During reset, use simpler navigation obs (before normalization values are computed)
        # During step, use normalized values
        if hasattr(self, 'timesteps') and self.timesteps > 0:
            # Step mode: use normalized navigation features
            normalized_velocity = self.velocity / self.target_speed
            normalized_distance_from_center = self.distance_from_center / self.max_distance_from_center
            normalized_angle = abs(self.angle / np.deg2rad(20))
            self.navigation_obs = np.array([
                self.throttle, 
                self.velocity, 
                normalized_velocity, 
                normalized_distance_from_center, 
                normalized_angle
            ])
        else:
            # Reset mode: use basic navigation features
            self.navigation_obs = np.array([
                self.throttle, 
                self.velocity, 
                self.previous_steer, 
                self.distance_from_center, 
                self.angle
            ])
        
        return [self.image_obs, self.navigation_obs]

    def _calculate_reward(self):
        """
        Calculate reward based on current state and check for termination conditions.
        
        Returns:
            tuple: (reward, done, collision_occurred, collision_intensity)
                - reward: calculated reward value
                - done: whether episode should terminate
                - collision_occurred: whether a collision was detected
                - collision_intensity: intensity of collision (0 if no collision)
        """
        done = False
        reward = 0.0
        collision_occurred = False
        collision_intensity = 0.0
        
        # Check for collisions FIRST - this is the most important termination condition
        # Re-check collision data directly from sensor right before checking (in case it was updated)
        if self.collision_obj is not None:
            current_collision_data = list(self.collision_obj.collision_data)  # Make a copy to avoid issues
        else:
            current_collision_data = []
        
        # Check if there are any collisions detected
        # Also check if collision history has grown (new collisions since last check)
        collision_detected = len(current_collision_data) > 0
        
        # Additional check: if collision data list has grown, we have a new collision
        if hasattr(self, 'previous_collision_count'):
            if len(current_collision_data) > self.previous_collision_count:
                collision_detected = True
        else:
            self.previous_collision_count = 0
        
        if collision_detected:
            collision_occurred = True
            done = True
            # Calculate collision intensity (sum of all collision impulses)
            collision_intensity = sum(current_collision_data)
            # Apply big penalty for collision
            collision_penalty = -50.0 - min(collision_intensity * 0.1, 50.0)
            reward = collision_penalty
            print(f"[COLLISION DETECTED] Timestep: {self.timesteps}, Intensity: {collision_intensity:.2f}, "
                  f"Penalty: {collision_penalty:.2f}, Collision count: {len(current_collision_data)}")
            print(f"  -> Collision values: {current_collision_data}")
            # Update previous collision count
            self.previous_collision_count = len(current_collision_data)
        elif self.distance_from_center > self.max_distance_from_center:
            # Reset only if too far from lane center
            done = True
            reward = -10.0
            print(f"[RESET] Too far from lane center: {self.distance_from_center:.2f} m > {self.max_distance_from_center:.2f} m")
            print(f"  -> Timestep: {self.timesteps}, Waypoint index: {self.current_waypoint_index}, "
                  f"Velocity: {self.velocity:.2f} km/h, Angle: {np.degrees(self.angle):.2f}°")
        
        # If episode is already done (collision or lane deviation), return early
        if done:
            return reward, done, collision_occurred, collision_intensity
        
        # Reward components (all normalized to [0, 1])
        # 1. Centering factor: 1.0 when centered, decreases as distance increases
        centering_factor = max(1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)
        
        # 2. Angle alignment factor: heavily penalize large deviation angles
        # Use exponential penalty for larger angles to strongly discourage misalignment
        # Use absolute value of angle since we penalize deviation regardless of direction (left/right)
        # Check for NaN/Inf in angle before using it
        if not np.isfinite(self.angle):
            angle_factor = 0.0  # Default to neutral if angle is invalid
        else:
            # Use absolute value of angle for reward calculation
            abs_angle = abs(self.angle)  # Already in radians, range [-π/2, π/2]
            abs_angle_deg = np.degrees(abs_angle)  # Convert to degrees for easier threshold comparison
            # Clamp to reasonable range to prevent extreme values (max 90° since angle is clamped to ±90°)
            abs_angle_deg = min(abs_angle_deg, 90.0)
            
            if abs_angle_deg <= 5.0:
                # Small angles (0-5°): full reward, linear decrease
                angle_factor = 1.0 - (abs_angle_deg / 5.0) * 0.2  # 0.8 to 1.0
            elif abs_angle_deg <= 15.0:
                # Medium angles (5-15°): significant penalty
                angle_factor = 0.8 - ((abs_angle_deg - 5.0) / 10.0) * 0.6  # 0.2 to 0.8
            elif abs_angle_deg <= 45.0:
                # Large angles (15-45°): heavy penalty
                angle_factor = 0.2 - ((abs_angle_deg - 15.0) / 30.0) * 0.3  # -0.1 to 0.2
            else:
                # Very large angles (>45°): severe negative penalty
                # Exponential penalty for angles > 45° (up to 90°)
                excess_angle = abs_angle_deg - 45.0
                angle_factor = -0.1 - (excess_angle / 45.0) * 0.5  # Can go down to -0.6 for 90°
            
            # Clamp angle_factor to reasonable range
            angle_factor = max(-1.0, min(1.0, angle_factor))
        
        # Final check for NaN/Inf
        if not np.isfinite(angle_factor):
            angle_factor = 0.0
        
        # 3. Speed factor: reward for maintaining target speed
        if self.velocity < self.min_speed:
            speed_factor = self.velocity / self.min_speed  # Penalize low speed
        elif self.velocity > self.target_speed:
            # Penalize overspeed, but less harshly
            speed_factor = max(1.0 - (self.velocity - self.target_speed) / (self.max_speed - self.target_speed), 0.0)
        else:
            speed_factor = 1.0  # Reward for being in target range
        
        # 4. Forward progress reward: reward for advancing along the route
        # Calculate progress since last step
        if hasattr(self, 'previous_waypoint_index'):
            progress = max(0, self.current_waypoint_index - self.previous_waypoint_index)
        else:
            progress = 0
        progress_factor = min(progress / 5.0, 1.0)  # Normalize progress (max reward for 5 waypoints)
        
        # 5. Base survival reward: small positive reward for staying alive
        survival_reward = 0.1
        
        # Combined reward: weighted sum of all factors
        # Prioritize centering and alignment, but also reward speed and progress
        # Increased weight on angle_factor to make it more important
        reward = (
            survival_reward +  # Base reward for staying alive
            0.25 * centering_factor +  # 25% weight on centering
            0.4 * angle_factor +       # 40% weight on alignment (increased from 30%)
            0.2 * speed_factor +       # 20% weight on speed
            0.05 * progress_factor     # 5% weight on forward progress (reduced)
        )
        
        # Scale to make rewards more meaningful (multiply by 2)
        reward = reward * 2.0
        
        # Clamp reward to prevent extreme values that could cause NaN in training
        # This prevents gradient explosion from extreme rewards
        reward = np.clip(reward, -10.0, 10.0)
        
        # Final check for NaN/Inf
        if not np.isfinite(reward):
            reward = 0.0  # Default to zero if reward is invalid
        
        return reward, done, collision_occurred, collision_intensity

    # A step function is used for taking inputs generated by neural net.
    def step(self, action_idx):
        try:

            self.timesteps+=1
            self.fresh_start = False

            # Velocity of the vehicle
            velocity = self.vehicle.get_velocity()
            self.velocity = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
            
            # Calculate angle BEFORE applying action (for immediate correction)
            # Calculate angle between car's heading direction and lane direction
            # This represents how much the car needs to turn to align with the lane
            if hasattr(self, 'route_waypoints') and self.route_waypoints and hasattr(self, 'current_waypoint_index'):
                try:
                    current_wp = self.route_waypoints[self.current_waypoint_index % len(self.route_waypoints)]
                    
                    # Get car's current transform (heading direction)
                    vehicle_transform = self.vehicle.get_transform()
                    car_forward = vehicle_transform.rotation.get_forward_vector()
                    car_forward_2d = np.array([car_forward.x, car_forward.y])
                    car_forward_norm = np.linalg.norm(car_forward_2d)
                    if car_forward_norm > 1e-6:
                        car_forward_2d = car_forward_2d / car_forward_norm
                    else:
                        car_forward_2d = np.array([1.0, 0.0])  # Default forward
                    
                    # Get lane direction from waypoint (where the lane is pointing)
                    lane_forward = current_wp.transform.rotation.get_forward_vector()
                    lane_forward_2d = np.array([lane_forward.x, lane_forward.y])
                    lane_forward_norm = np.linalg.norm(lane_forward_2d)
                    if lane_forward_norm > 1e-6:
                        lane_forward_2d = lane_forward_2d / lane_forward_norm
                    else:
                        lane_forward_2d = np.array([1.0, 0.0])  # Default forward
                    
                    # Calculate angle between car heading and lane direction using atan2
                    # This gives us the signed angle from car direction to lane direction
                    car_angle = np.arctan2(car_forward_2d[1], car_forward_2d[0])
                    lane_angle = np.arctan2(lane_forward_2d[1], lane_forward_2d[0])
                    
                    # Calculate signed angle difference
                    angle_diff = lane_angle - car_angle
                    
                    # Normalize to [-π, π] range
                    while angle_diff > np.pi:
                        angle_diff -= 2 * np.pi
                    while angle_diff < -np.pi:
                        angle_diff += 2 * np.pi
                    
                    # For forward movement, we want angle in [-90°, 90°] range
                    # If the angle is > 90° or < -90°, it means the car is pointing backwards
                    # relative to the lane, so we take the complement
                    if angle_diff > np.pi / 2:
                        angle_diff = np.pi - angle_diff
                    elif angle_diff < -np.pi / 2:
                        angle_diff = -np.pi - angle_diff
                    
                    current_angle = angle_diff
                    
                    # Clamp to -90° to 90° range (safety check)
                    current_angle = np.clip(current_angle, -np.pi/2, np.pi/2)
                    
                    # Check for NaN or Inf values
                    if not np.isfinite(current_angle):
                        current_angle = self.angle if hasattr(self, 'angle') and np.isfinite(self.angle) else 0.0
                        
                except Exception as e:
                    # Fallback to previous angle or zero
                    current_angle = self.angle if hasattr(self, 'angle') and np.isfinite(self.angle) else 0.0
            else:
                current_angle = self.angle if hasattr(self, 'angle') and np.isfinite(self.angle) else 0.0
            
            # Calculate distance from center BEFORE applying action (for lane correction)
            # This uses the current position to help correct steering
            if hasattr(self, 'route_waypoints') and self.route_waypoints and hasattr(self, 'current_waypoint_index'):
                try:
                    current_wp = self.route_waypoints[self.current_waypoint_index % len(self.route_waypoints)]
                    next_wp = self.route_waypoints[(self.current_waypoint_index + 1) % len(self.route_waypoints)]
                    current_distance = self.distance_to_line(
                        self.vector(current_wp.transform.location),
                        self.vector(next_wp.transform.location),
                        self.vector(self.vehicle.get_location())
                    )
                except:
                    current_distance = self.distance_from_center if hasattr(self, 'distance_from_center') else 0.0
            else:
                current_distance = self.distance_from_center if hasattr(self, 'distance_from_center') else 0.0
            
            # Action fron action space for contolling the vehicle with a discrete action
            if self.continous_action_space:
                steer = float(action_idx[0])
                steer = max(min(steer, 1.0), -1.0)
                throttle = float((action_idx[1] + 1.0)/2)
                throttle = max(min(throttle, 1.0), 0.0)
                
                # Ensure minimum throttle to keep car moving forward
                # If throttle is too low, use a minimum value to prevent stopping
                if throttle < 0.3:
                    throttle = 0.3  # Minimum throttle to maintain forward movement
                
                # Add steering decay to prevent accumulation of steering bias
                # Decay previous APPLIED steering (not raw action) towards zero
                # Use stronger decay (30% instead of 50%) to prevent bias accumulation
                decayed_previous_steer = self.previous_steer * 0.3  # Decay by 70% (keep only 30%)
                
                # Always apply angle correction to prevent angle from growing
                # This is essential to prevent the car from spiraling out of control
                # Use STRONGER correction since angle keeps growing
                # Use CURRENT angle (calculated before action) for immediate correction
                angle_correction = 0.0
                if abs(current_angle) > np.deg2rad(2):  # Correct if angle > 2 degrees (lower threshold)
                    # If angle is positive (pointing right), steer left (negative) to correct
                    # If angle is negative (pointing left), steer right (positive) to correct
                    # Use stronger correction: scale by angle with max of 0.4 (increased from 0.2)
                    # Use exponential scaling for larger angles to prevent runaway
                    angle_magnitude = abs(current_angle)
                    if angle_magnitude > np.deg2rad(10):
                        # For large angles, use stronger correction
                        correction_strength = 0.4 + 0.2 * (angle_magnitude - np.deg2rad(10)) / np.deg2rad(20)
                        correction_strength = min(correction_strength, 0.6)  # Cap at 0.6
                    else:
                        # For small angles, use proportional correction
                        correction_strength = 0.3 * (angle_magnitude / np.deg2rad(10))
                    
                    angle_correction = -np.sign(current_angle) * correction_strength
                
                # Optional additional corrections (lane-based)
                lane_correction = 0.0
                if self.enable_auto_correction and abs(current_distance) > 1.0:
                    lane_correction = -np.sign(current_distance) * min(abs(current_distance) / self.max_distance_from_center, 0.1)
                
                # Combine corrections: angle correction is always active, lane is optional
                total_correction = angle_correction + lane_correction
                
                # Apply steering with decay and correction
                # Prioritize correction over agent action when angle is large
                if abs(current_angle) > np.deg2rad(10):
                    # For large angles, use 80% correction + 20% agent action
                    corrected_steer = steer * 0.2 + total_correction * 0.8
                    smoothed_steer = decayed_previous_steer*0.2 + corrected_steer*0.8
                else:
                    # For small angles, balance agent action and correction
                    corrected_steer = steer + total_correction
                    smoothed_steer = decayed_previous_steer*0.3 + corrected_steer*0.7
                
                # Limit steering to prevent extreme turns
                smoothed_steer = max(min(smoothed_steer, 0.8), -0.8)
                
                # Use more of the new throttle (less filtering) to allow faster response
                smoothed_throttle = self.throttle*0.5 + throttle*0.5
                
                self.vehicle.apply_control(carla.VehicleControl(
                    steer=smoothed_steer, 
                    throttle=smoothed_throttle
                ))
                # Store the ACTUAL applied steering (smoothed), not raw action
                # This ensures decay works on what was actually applied
                self.previous_steer = smoothed_steer
                self.throttle = throttle
            else:
                steer = self.action_space[action_idx]
                if self.velocity < 20.0:
                    self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1, throttle=1.0))
                else:
                    self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1))
                self.previous_steer = steer
                self.throttle = 1.0
            
            # Traffic Light state
            if self.vehicle.is_at_traffic_light():
                traffic_light = self.vehicle.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    traffic_light.set_state(carla.TrafficLightState.Green)

            # Update collision history - check directly from collision sensor
            # The collision_data is a list that accumulates collision events
            if self.collision_obj is not None:
                self.collision_history = self.collision_obj.collision_data
            else:
                self.collision_history = []

            # Rotation of the vehicle in correlation to the map/lane
            self.rotation = self.vehicle.get_transform().rotation.yaw

            # Location of the car
            self.location = self.vehicle.get_location()


            #transform = self.vehicle.get_transform()
            # Keep track of closest waypoint on the route
            waypoint_index = self.current_waypoint_index
            for _ in range(len(self.route_waypoints)):
                # Check if we passed the next waypoint along the route
                next_waypoint_index = waypoint_index + 1
                wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
                dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2],self.vector(self.location - wp.transform.location)[:2])
                if dot > 0.0:
                    waypoint_index += 1
                else:
                    break

            self.current_waypoint_index = waypoint_index
            # Calculate deviation from center of the lane
            self.current_waypoint = self.route_waypoints[ self.current_waypoint_index    % len(self.route_waypoints)]
            self.next_waypoint = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]
            self.distance_from_center = self.distance_to_line(self.vector(self.current_waypoint.transform.location),self.vector(self.next_waypoint.transform.location),self.vector(self.location))
            self.center_lane_deviation += self.distance_from_center

            # Store the angle we calculated earlier (for use in next step and rewards)
            # The angle was already calculated before applying the action
            self.angle = current_angle

             # Update checkpoint for training
            if not self.fresh_start:
                if self.checkpoint_frequency is not None:
                    self.checkpoint_waypoint_index = (self.current_waypoint_index // self.checkpoint_frequency) * self.checkpoint_frequency

            
            # Calculate reward and check termination conditions
            reward, done, collision_occurred, collision_intensity = self._calculate_reward()
            
            # Store previous waypoint index for progress calculation
            self.previous_waypoint_index = self.current_waypoint_index
            
            # Update previous collision count for next step (to detect new collisions)
            if self.collision_obj is not None:
                self.previous_collision_count = len(self.collision_obj.collision_data)
            else:
                self.previous_collision_count = 0

            # Only reset on collision or lane deviation (already checked above)
            # Also reset if episode is too long (max timesteps) or route completed
            if not done:  # Only check these if not already done
                if self.timesteps >= 7500:
                    done = True
                    print(f"[RESET] Maximum timesteps reached: {self.timesteps}")
                    print(f"  -> Distance from center: {self.distance_from_center:.2f} m, "
                          f"Waypoint index: {self.current_waypoint_index}/{len(self.route_waypoints)}")
                elif self.current_waypoint_index >= len(self.route_waypoints) - 2:
                    done = True
                    self.fresh_start = True
                    print(f"[RESET] Route completed at waypoint {self.current_waypoint_index}/{len(self.route_waypoints)}")
                    print(f"  -> Distance from center: {self.distance_from_center:.2f} m, "
                          f"Timesteps: {self.timesteps}")
                    if self.checkpoint_frequency is not None:
                        if self.checkpoint_frequency < self.total_distance//2:
                            self.checkpoint_frequency += 2
                        else:
                            self.checkpoint_frequency = None
                            self.checkpoint_waypoint_index = 0

            # Get observation using the observation function
            observation = self._get_observation()
            
            # Remove everything that has been spawned in the env
            if done:
                self.center_lane_deviation = self.center_lane_deviation / self.timesteps
                self.distance_covered = abs(self.current_waypoint_index - self.checkpoint_waypoint_index)
                
                for sensor in self.sensor_list:
                    sensor.destroy()
                
                self.remove_sensors()
                
                for actor in self.actor_list:
                    actor.destroy()
            
            # Return info with collision status (info[4] = collision_occurred)
            return observation, reward, done, [self.distance_covered, self.center_lane_deviation, self.distance_from_center, self.angle, collision_occurred]

        except Exception as e:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()
            raise e



# -------------------------------------------------
# Creating and Spawning Pedestrians in our world |
# -------------------------------------------------

    # Walkers are to be included in the simulation yet!
    def create_pedestrians(self):
        try:

            # Our code for this method has been broken into 3 sections.

            # 1. Getting the available spawn points in  our world.
            # Random Spawn locations for the walker
            walker_spawn_points = []
            for i in range(NUMBER_OF_PEDESTRIAN):
                spawn_point_ = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if (loc != None):
                    spawn_point_.location = loc
                    walker_spawn_points.append(spawn_point_)

            # 2. We spawn the walker actor and ai controller
            # Also set their respective attributes
            for spawn_point_ in walker_spawn_points:
                walker_bp = random.choice(
                    self.blueprint_library.filter('walker.pedestrian.*'))
                walker_controller_bp = self.blueprint_library.find(
                    'controller.ai.walker')
                # Walkers are made visible in the simulation
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                # They're all walking not running on their recommended speed
                if walker_bp.has_attribute('speed'):
                    walker_bp.set_attribute(
                        'speed', (walker_bp.get_attribute('speed').recommended_values[1]))
                else:
                    walker_bp.set_attribute('speed', 0.0)
                walker = self.world.try_spawn_actor(walker_bp, spawn_point_)
                if walker is not None:
                    walker_controller = self.world.spawn_actor(
                        walker_controller_bp, carla.Transform(), walker)
                    self.walker_list.append(walker_controller.id)
                    self.walker_list.append(walker.id)
            all_actors = self.world.get_actors(self.walker_list)

            # set how many pedestrians can cross the road
            #self.world.set_pedestrians_cross_factor(0.0)
            # 3. Starting the motion of our pedestrians
            for i in range(0, len(self.walker_list), 2):
                # start walker
                all_actors[i].start()
            # set walk to random point
                all_actors[i].go_to_location(
                    self.world.get_random_location_from_navigation())

        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.walker_list])


# ---------------------------------------------------
# Creating and Spawning other vehciles in our world|
# ---------------------------------------------------


    def set_other_vehicles(self):
        try:
            # NPC vehicles generated and set to autopilot
            # One simple for loop for creating x number of vehicles and spawing them into the world
            for _ in range(0, NUMBER_OF_VEHICLES):
                spawn_point = random.choice(self.map.get_spawn_points())
                bp_vehicle = random.choice(self.blueprint_library.filter('vehicle'))
                other_vehicle = self.world.try_spawn_actor(
                    bp_vehicle, spawn_point)
                if other_vehicle is not None:
                    other_vehicle.set_autopilot(True)
                    self.actor_list.append(other_vehicle)
            print("NPC vehicles have been generated in autopilot mode.")
        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.actor_list])


# ----------------------------------------------------------------
# Extra very important methods: their names explain their purpose|
# ----------------------------------------------------------------

    # Setter for changing the town on the server.
    def change_town(self, new_town):
        self.world = self.client.load_world(new_town)


    # Getter for fetching the current state of the world that simulator is in.
    def get_world(self) -> object:
        return self.world


    # Getter for fetching blueprint library of the simulator.
    def get_blueprint_library(self) -> object:
        return self.world.get_blueprint_library()


    # Action space of our vehicle. It can make eight unique actions.
    # Continuous actions are broken into discrete here!
    def angle_diff(self, v0, v1):
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        if angle > np.pi: angle -= 2 * np.pi
        elif angle <= -np.pi: angle += 2 * np.pi
        return angle


    def distance_to_line(self, A, B, p):
        num   = np.linalg.norm(np.cross(B - A, A - p))
        denom = np.linalg.norm(B - A)
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom


    def vector(self, v):
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])


    def get_discrete_action_space(self):
        action_space = \
            np.array([
            -0.50,
            -0.30,
            -0.10,
            0.0,
            0.10,
            0.30,
            0.50
            ])
        return action_space

    # Main vehicle blueprint method
    # It picks a random color for the vehicle everytime this method is called
    def get_vehicle(self, vehicle_name):
        blueprint = self.blueprint_library.filter(vehicle_name)[0]
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        return blueprint


    # Spawn the vehicle in the environment
    def set_vehicle(self, vehicle_bp, spawn_points):
        # Main vehicle spawned into the env
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)


    # Clean up method
    def remove_sensors(self):
        self.camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None
        self.env_camera_obj = None
        self.front_camera = None
        self.collision_history = None
        self.wrong_maneuver = None


