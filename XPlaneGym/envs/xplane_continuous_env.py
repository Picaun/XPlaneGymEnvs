"""
XPlaneContinuousEnv - X-Plane Environment with Continuous Action Space

This environment uses a continuous action space, suitable for reinforcement learning algorithms 
based on continuous actions such as PPO, SAC, etc.
It provides more precise control capabilities, suitable for implementing complex flight control tasks.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union
import gymnasium as gym
from gymnasium import spaces
from .xplane_env import XPlaneEnv


class XPlaneContinuousEnv(XPlaneEnv):
    """X-Plane Environment with Continuous Action Space
    
    Provides continuous control inputs, suitable for reinforcement learning algorithms
    that support continuous action spaces such as PPO, SAC, etc.
    Action space includes pitch control, roll control, rudder control, and throttle dimensions.
    """
    
    def __init__(self, 
                 ip: str = '127.0.0.1',
                 port: int = 49000,
                 timeout: float = 1.0,
                 pause_delay: float = 0.05,
                 starting_latitude: float = 37.558,
                 starting_longitude: float = 126.790,
                 starting_altitude: float = 3000.0,
                 starting_velocity: float = 100.0,
                 starting_pitch_range: float = 10.0,
                 starting_roll_range: float = 20.0,
                 random_desired_state: bool = True,
                 desired_pitch_range: float = 5.0,
                 desired_roll_range: float = 10.0,
                 render_mode: Optional[str] = None,
                 action_dim: int = 4):
        """Initialize X-Plane environment with continuous action space
        
        Args:
            ip: IP address where X-Plane is running
            port: X-Plane UDP communication port
            timeout: Communication timeout in seconds
            pause_delay: Pause delay time in seconds
            starting_latitude: Initial latitude (degrees)
            starting_longitude: Initial longitude (degrees)
            starting_altitude: Initial altitude (meters)
            starting_velocity: Initial velocity (meters/second)
            starting_pitch_range: Initial pitch angle random range (degrees)
            starting_roll_range: Initial roll angle random range (degrees)
            random_desired_state: Whether to use random target state
            desired_pitch_range: Target pitch angle random range (degrees)
            desired_roll_range: Target roll angle random range (degrees)
            render_mode: Render mode
            action_dim: Action space dimension
        """
        # Call parent constructor, specifying continuous action space
        super().__init__(
            ip=ip,
            port=port,
            timeout=timeout,
            pause_delay=pause_delay,
            starting_latitude=starting_latitude,
            starting_longitude=starting_longitude,
            starting_altitude=starting_altitude,
            starting_velocity=starting_velocity,
            starting_pitch_range=starting_pitch_range,
            starting_roll_range=starting_roll_range,
            random_desired_state=random_desired_state,
            desired_pitch_range=desired_pitch_range,
            desired_roll_range=desired_roll_range,
            render_mode=render_mode,
            continuous_actions=True  # Specify continuous action space
        )
        
        # Customize continuous action space, supporting 2-4 dimensions
        if action_dim == 2:
            # Pitch and roll control only
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0]),
                high=np.array([1.0, 1.0]),
                dtype=np.float32
            )
        elif action_dim == 3:
            # Pitch, roll and rudder control
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0, -1.0]),
                high=np.array([1.0, 1.0, 1.0]),
                dtype=np.float32
            )
        else:
            # Pitch, roll, rudder control and throttle
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0, -1.0, 0.0]),
                high=np.array([1.0, 1.0, 1.0, 1.0]),
                dtype=np.float32
            )
        
        # Additional parameters for continuous action control
        self.action_smoothing = 0.7  # Action smoothing factor, higher value means smoother action changes
        self.previous_action = np.zeros(self.action_space.shape[0])  # Previous action
        self.action_dimension = action_dim  # Action space dimension
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute a continuous action and return the next state, reward, and whether the episode is done
        
        Override parent method to add action smoothing
        
        Args:
            action: Continuous action array
            
        Returns:
            observation: New observation state
            reward: Reward received after executing the action
            terminated: Whether terminated (completed or failed)
            truncated: Whether truncated due to exceeding max steps or other reasons
            info: Additional information
        """
        # Smooth the action to reduce drastic changes
        smoothed_action = self._smooth_action(action)
        
        # Call parent step method
        observation, reward, terminated, truncated, info = super().step(smoothed_action)
        
        # Record current action for next step smoothing
        self.previous_action = smoothed_action
        
        # Add additional information
        info["raw_action"] = action
        info["smoothed_action"] = smoothed_action
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state
        
        Override parent method, also reset action smoothing state
        
        Args:
            seed: Random seed
            options: Reset options
            
        Returns:
            observation: Initial observation state
            info: Additional information
        """
        # Reset action smoothing state
        self.previous_action = np.zeros(self.action_space.shape[0])
        
        # Call parent reset method
        return super().reset(seed=seed, options=options)
    
    def _smooth_action(self, action: np.ndarray) -> np.ndarray:
        """Smooth action to avoid drastic changes in control inputs
        
        Args:
            action: Original action
            
        Returns:
            Smoothed action
        """
        # Ensure action dimensions match
        if len(action) != len(self.previous_action):
            # If dimensions don't match, use original action
            return action
        
        # Limit action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Apply smoothing factor
        smoothed_action = self.action_smoothing * self.previous_action + (1 - self.action_smoothing) * action
        
        return smoothed_action
    
    def _action_to_control(self, action: np.ndarray) -> Dict[str, float]:
        """Convert continuous action to control input
        
        Override parent method to provide more fine-grained control mapping
        
        Args:
            action: Continuous action array
            
        Returns:
            Control input dictionary
        """
        # Get attitude data for adaptive control
        attitude = self.client.get_attitude()
        roll = attitude["roll"]
        pitch = attitude["pitch"]
        
        # Calculate adaptive factor based on attitude deviation
        adaptive_factor = self._compute_adaptive_factor(roll, pitch)
        
        # Default control input
        control = {
            "lon_control": 0.0,  # Pitch control
            "lat_control": 0.0,  # Roll control
            "rudder_control": 0.0,  # Rudder control
            "throttle": 0.5  # Throttle
        }
        
        # Set control values based on action dimension
        if len(action) >= 1:
            # Pitch control (first dimension in action space)
            control["lon_control"] = float(action[0]) * adaptive_factor
        
        if len(action) >= 2:
            # Roll control (second dimension in action space)
            control["lat_control"] = float(action[1]) * adaptive_factor
        
        if len(action) >= 3:
            # Rudder control (third dimension in action space)
            control["rudder_control"] = float(action[2]) * adaptive_factor
        
        if len(action) >= 4:
            # Throttle control (fourth dimension in action space)
            # Throttle uses absolute value, no need for adaptive factor
            control["throttle"] = float(action[3])
        
        return control
    
    def _compute_adaptive_factor(self, roll: float, pitch: float) -> float:
        """Calculate adaptive control factor
        
        Calculate an adjustment factor based on current attitude deviation,
        used to scale control input intensity. Use larger control inputs when
        attitude deviation is large, and smaller control inputs when attitude
        is close to desired value.
        
        Args:
            roll: Current roll angle (degrees)
            pitch: Current pitch angle (degrees)
            
        Returns:
            Adaptive control factor
        """
        # Calculate deviation from target attitude
        if self.random_desired_state:
            roll_error = abs(self.desired_state["roll"] - roll)
            pitch_error = abs(self.desired_state["pitch"] - pitch)
        else:
            roll_error = abs(roll)
            pitch_error = abs(pitch)
        
        # Take maximum deviation
        max_error = max(roll_error, pitch_error)
        
        # Calculate adaptive factor based on deviation
        if max_error < 1.0:
            return 0.3  # Very close to target, use small control input
        elif max_error < 3.0:
            return 0.4  # Close to target, use moderate control input
        elif max_error < 10.0:
            return 0.6  # Medium deviation, use larger control input
        elif max_error < 30.0:
            return 0.8  # Large deviation, use large control input
        else:
            return 1.0  # Far from target, use maximum control input
    
    def _compute_reward(self, action: np.ndarray, observation: np.ndarray) -> Tuple[float, bool]:
        """Calculate reward and termination for continuous action space
        
        Override parent method to provide more appropriate reward calculation for continuous control
        
        Args:
            action: Executed continuous action
            observation: Current observation state
            
        Returns:
            reward: Reward value
            terminated: Whether terminated
        """
        # Basic reward calculation (using parent method)
        reward, terminated = super()._compute_reward(action, observation)
        
        # Special adjustments for continuous environment
        
        # 1. Control smoothness reward: encourage smooth control inputs, penalize drastic control changes
        if len(self.previous_action) == len(action):
            action_change = np.sum(np.square(action - self.previous_action))
            smoothness_penalty = 0.1 * action_change  # Penalty for control changes
            reward -= min(0.2, smoothness_penalty)  # Limit maximum penalty
        
        # 2. Attitude stability reward: provide extra reward when attitude is close to target and stable
        roll_error = abs(observation[0])
        pitch_error = abs(observation[1])
        
        if roll_error < 2.0 and pitch_error < 2.0:
            # Attitude very close to target
            roll_rate = abs(observation[3])  # Roll rate
            pitch_rate = abs(observation[4])  # Pitch rate
            
            if roll_rate < 0.05 and pitch_rate < 0.05:
                # Attitude rates very small, indicating stable attitude
                reward += 0.3  # Stability reward
        
        return reward, terminated
    
    def _get_observation(self) -> np.ndarray:
        """Get current environment observation state
        
        Override parent method to provide richer state information for continuous control
        
        Returns:
            Observation state array
        """
        # Get basic observation state (using parent method)
        base_observation = super()._get_observation()
        
        # Continuous control environment could consider adding more state information
        # For example, add previous action information to help learn action continuity
        # Here we keep the same state representation as the parent class for compatibility
        
        return base_observation 