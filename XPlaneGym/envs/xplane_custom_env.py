import numpy as np
from typing import Tuple, Dict, Optional, Union
import gymnasium as gym
from gymnasium import spaces
from .xplane_env import XPlaneEnv


class XPlaneCustomEnv(XPlaneEnv):
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
                 n_actions: int = 9):
        """Initialize X-Plane environment with discrete action space
        
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
            n_actions: Size of discrete action space
        """
        # Call parent constructor, specifying discrete action space
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
            continuous_actions=False  # Specify discrete action space
        )
        
        # Customize discrete action space size
        self.action_space = spaces.Discrete(n_actions)
        
        # Extended action mapping dictionary
        self.action_map = {
            0: "no_action",       # No action
            1: "pitch_up",        # Pitch up
            2: "pitch_down",      # Pitch down
            3: "roll_right",      # Roll right
            4: "roll_left",       # Roll left
            5: "rudder_right",    # Rudder right
            6: "rudder_left",     # Rudder left
            7: "throttle_up",     # Throttle increase
            8: "throttle_down"    # Throttle decrease
        }
        
        # Additional parameters for DQN and similar algorithms
        self.reward_discount_factor = 0.99  # Reward discount factor
        self.control_intensity = 0.3  # Control input intensity
    
    def _action_to_control(self, action: int) -> Dict[str, float]:
        """Convert discrete action to control input
        
        Override parent method to provide more fine-grained control mapping
        
        Args:
            action: Discrete action index
            
        Returns:
            Control input dictionary
        """
        # Default control input
        control = {"lon_control": 0.0, "lat_control": 0.0, "rudder_control": 0.0, "throttle": 0.5}
        
        # Ensure action is within valid range
        action = int(action) % self.action_space.n
        
        # Get latest attitude information for dynamic control intensity adjustment
        attitude = self.client.get_attitude()
        roll = abs(attitude["roll"])
        pitch = abs(attitude["pitch"])
        
        # Adjust control intensity based on current attitude deviation
        intensity = self._get_control_intensity(roll, pitch)
        
        # Set control values based on action index
        if action == 0:  # No action
            pass
        elif action == 1:  # Pitch up
            control["lon_control"] = intensity
        elif action == 2:  # Pitch down
            control["lon_control"] = -intensity
        elif action == 3:  # Roll right
            control["lat_control"] = intensity
        elif action == 4:  # Roll left
            control["lat_control"] = -intensity
        elif action == 5:  # Rudder right
            control["rudder_control"] = intensity
        elif action == 6:  # Rudder left
            control["rudder_control"] = -intensity
        elif action == 7:  # Throttle increase
            prev_throttle = self.client.get_dref("sim/cockpit2/engine/actuators/throttle_jet_rev_ratio_all")
            control["throttle"] = min(1.0, prev_throttle + 0.1)
        elif action == 8:  # Throttle decrease
            prev_throttle = self.client.get_dref("sim/cockpit2/engine/actuators/throttle_jet_rev_ratio_all")
            control["throttle"] = max(0.0, prev_throttle - 0.1)
        
        return control
    
    def _get_control_intensity(self, roll: float, pitch: float) -> float:
        """Calculate control intensity based on current attitude deviation
        
        Args:
            roll: Current roll angle deviation (degrees)
            pitch: Current pitch angle deviation (degrees)
            
        Returns:
            Control intensity, range [0.05, 1.0]
        """
        # Take the maximum of attitude deviations
        max_deviation = max(roll, pitch)
        
        # Return different control intensities based on deviation magnitude
        if max_deviation < 1.0:
            return 0.05  # Very small correction
        elif max_deviation < 3.0:
            return 0.1  # Small correction
        elif max_deviation < 5.0:
            return 0.15  # Medium correction
        elif max_deviation < 10.0:
            return 0.25  # Large correction
        elif max_deviation < 20.0:
            return 0.4  # Very large correction
        elif max_deviation < 40.0:
            return 0.6  # Extreme correction
        else:
            return 0.8  # Maximum correction
    
    def _compute_reward(self, action: int, observation: np.ndarray) -> Tuple[float, bool]:
        """Calculate reward and termination for discrete action space
        
        Override parent method to provide more appropriate reward calculation for discrete control
        
        Args:
            action: Executed discrete action
            observation: Current observation state
            
        Returns:
            reward: Reward value
            terminated: Whether terminated
        """
        # Basic reward calculation (using parent method)
        reward, terminated = super()._compute_reward(action, observation)
        
        # Special adjustments for discrete environment
        # Extra reward for no action when attitude is close to desired state
        if action == 0 and abs(observation[0]) < 2.0 and abs(observation[1]) < 2.0:
            reward += 0.2  # Encourage staying still when attitude is stable
        
        # Slight penalty for too frequent control inputs
        if action != 0:
            reward -= 0.05  # Control input cost
        
        return reward, terminated
    
