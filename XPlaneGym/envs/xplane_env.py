"""
XPlaneGym - Base Reinforcement Learning Environment for X-Plane Flight Simulator

This class provides a base environment for X-Plane that conforms to the OpenAI Gym interface specification,
containing basic environment setup, state transitions, and reward calculation logic.
Subclasses can inherit and extend this class to implement specific learning tasks.
"""

import time
import math
import numpy as np
from typing import Tuple, Dict, List, Any, Optional, Union
import gymnasium as gym
from gymnasium import spaces
import sys
import os
from XPlaneGym.xplane_connect import XPlaneConnect


class XPlaneEnv(gym.Env):
    """Base Reinforcement Learning Environment for X-Plane Flight Simulator
    
    Implements an environment conforming to the Gym interface specification,
    handles communication with X-Plane, and provides standard step, reset,
    and render interfaces.
    """
    
    metadata = {'render_modes': ['human']}
    
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
                 continuous_actions: bool = False):
        """Initialize X-Plane environment
        
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
            continuous_actions: Whether to use continuous action space
        """
        super().__init__()
        
        # Initialize X-Plane connection
        self.client = XPlaneConnect(ip=ip, port=port, timeout=timeout)
        
        # Environment parameters
        self.pause_delay = pause_delay
        self.starting_latitude = starting_latitude
        self.starting_longitude = starting_longitude
        self.starting_altitude = starting_altitude
        self.starting_velocity = starting_velocity
        self.starting_pitch_range = starting_pitch_range
        self.starting_roll_range = starting_roll_range
        self.random_desired_state = random_desired_state
        self.desired_pitch_range = desired_pitch_range
        self.desired_roll_range = desired_roll_range
        self.render_mode = render_mode
        self.continuous_actions = continuous_actions
        
        # Flight position and state
        self.starting_position = [starting_latitude, starting_longitude, starting_altitude]  # latitude, longitude, altitude
        self.starting_orientation = [0.0, 0.0, 0.0]  # roll, pitch, yaw
        self.desired_state = {"roll": 0.0, "pitch": 0.0}
        self.previous_state = None
        
        # Required DataRefs for environment state updates
        self.subscribed_drefs = [
            ("sim/flightmodel/position/latitude", 10),
            ("sim/flightmodel/position/longitude", 10),
            ("sim/flightmodel/position/elevation", 10),
            ("sim/flightmodel/position/phi", 10),  # roll angle
            ("sim/flightmodel/position/theta", 10),  # pitch angle
            ("sim/flightmodel/position/psi", 10),  # yaw angle
            ("sim/flightmodel/position/P", 10),  # roll rate
            ("sim/flightmodel/position/Q", 10),  # pitch rate
            ("sim/flightmodel/position/R", 10),  # yaw rate
            ("sim/flightmodel/position/alpha", 10),  # angle of attack
            ("sim/flightmodel/position/beta", 10),  # sideslip angle
            ("sim/flightmodel/position/y_agl", 10),  # height above ground
            ("sim/flightmodel2/misc/has_crashed", 10),  # whether crashed
        ]
        
        # State space (default implementation)
        # Subclasses can override this space to implement different state representations
        self.observation_space = spaces.Box(
            low=np.array([
                -180.0,   # roll error (deg)
                -90.0,    # pitch error (deg)
                0.0,      # heading (deg)
                -10.0,    # roll rate P (rad/s)
                -10.0,    # pitch rate Q (rad/s)
                -10.0,    # yaw rate R (rad/s)
                -45.0,    # angle of attack alpha (deg)
                -45.0,    # sideslip beta (deg)
                0.0,      # height AGL (m)
                -90.0,    # latitude (deg)
                -180.0,   # longitude (deg)
                -1000.0   # elevation MSL (m)
            ], dtype=np.float32),
            high=np.array([
                180.0,    # roll error (deg)
                90.0,     # pitch error (deg)
                360.0,    # heading (deg)
                10.0,     # roll rate P (rad/s)
                10.0,     # pitch rate Q (rad/s)
                10.0,     # yaw rate R (rad/s)
                45.0,     # angle of attack alpha (deg)
                45.0,     # sideslip beta (deg)
                100000.0, # height AGL (m)
                90.0,     # latitude (deg)
                180.0,    # longitude (deg)
                100000.0  # elevation MSL (m)
            ], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space (default implementation)
        # Base class provides two action space formats: discrete and continuous
        if continuous_actions:
            # Continuous action space: [pitch control, roll control, rudder control, throttle]
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0, -1.0, 0.0]),
                high=np.array([1.0, 1.0, 1.0, 1.0]),
                dtype=np.float32
            )
        else:
            # Discrete action space: 9 actions (no action, pitch up/down, roll left/right, rudder left/right, throttle up/down)
            self.action_space = spaces.Discrete(9)
            
            # Action mapping dictionary (discrete action indices)
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
        
        # Initialize connection
        self.client.subscribe_drefs(self.subscribed_drefs)
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute an action and return the next state, reward, and whether the episode is done
        
        Args:
            action: Action to execute, can be a discrete action index or continuous action array
            
        Returns:
            observation: New observation state
            reward: Reward received after executing the action
            terminated: Whether terminated (completed or failed)
            truncated: Whether truncated due to exceeding max steps or other reasons
            info: Additional information
        """
        # Convert action to control input
        ctrl = self._action_to_control(action)
        
        # Execute action
        self.client.pause_sim(False)  # Continue simulation
        self.client.send_ctrl(**ctrl)  # Send control input
        time.sleep(self.pause_delay)  # Wait for simulator to update state
        self.client.pause_sim(True)  # Pause simulation
        
        # Get new state
        observation = self._get_observation()
        
        # Calculate reward and whether terminated
        reward, terminated = self._compute_reward(action, observation)
        
        # Additional information
        info = {
            "position": self._get_position(),
            "action": action,
            "control": ctrl,
            "desired_state": self.desired_state
        }
        
        return observation, reward, terminated, False, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state
        
        Args:
            seed: Random seed
            options: Reset options
            
        Returns:
            observation: Initial observation state
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Random or preset initial attitude
        if options and "position" in options:
            # Use provided initial position
            position = options["position"]
            orientation = options.get("orientation", [0.0, 0.0, 0.0])
        else:
            # Use random initial position and attitude
            position, orientation = self._get_random_starting_position()
        
        # Random or preset target state
        if options and "desired_state" in options:
            self.desired_state = options["desired_state"]
        elif self.random_desired_state:
            self.desired_state = self._get_random_desired_state()
        else:
            self.desired_state = {"roll": 0.0, "pitch": 0.0}
        
        # Set aircraft position and attitude
        lat, lon, alt = self.starting_position
        roll, pitch, heading = orientation
        
        # Ensure paused
        self.client.pause_sim(True)
        
        # Set position
        self.client.send_position(
            lat=lat, 
            lon=lon, 
            alt=alt,
            phi=roll,  # roll angle
            theta=pitch,  # pitch angle
            psi=heading  # yaw angle
        )
        
        # Set initial velocity
        velocity = self._calculate_velocity(pitch, heading, self.starting_velocity)
        for dref, value in velocity.items():
            self.client.send_dref(dref, value)
        
        # Get initial observation state
        observation = self._get_observation()
        self.previous_state = observation
        
        # Additional information
        info = {
            "position": self._get_position(),
            "orientation": orientation,
            "desired_state": self.desired_state
        }
        
        return observation, info
    
    def render(self) -> None:
        """Render the environment
        
        In the X-Plane environment, rendering is done by the X-Plane simulator itself.
        This method can be used to display additional information or special views.
        """
        if self.render_mode == "human":
            # X-Plane is already rendering, additional display logic can be added here
            pass
    
    def close(self) -> None:
        """Close the environment, release resources"""
        # Stop data reception thread
        self.client.stop()
    
    def _action_to_control(self, action: Union[int, np.ndarray]) -> Dict[str, float]:
        """Convert action to control input
        
        Args:
            action: Action, can be a discrete action index or continuous action array
            
        Returns:
            Control input dictionary
        """
        if self.continuous_actions:
            # Continuous action space
            if len(action) >= 4:
                return {
                    "lon_control": float(action[0]),  # pitch control
                    "lat_control": float(action[1]),  # roll control
                    "rudder_control": float(action[2]),  # rudder control
                    "throttle": float(action[3])  # throttle
                }
            else:
                return {
                    "lon_control": float(action[0]) if len(action) > 0 else 0.0,
                    "lat_control": float(action[1]) if len(action) > 1 else 0.0,
                    "rudder_control": 0.0,
                    "throttle": 0.5
                }
        else:
            # Discrete action space
            control = {"lon_control": 0.0, "lat_control": 0.0, "rudder_control": 0.0, "throttle": 0.5}
            
            # Set control values based on action index
            if action == 0:  # No action
                pass
            elif action == 1:  # Pitch up
                control["lon_control"] = 0.3
            elif action == 2:  # Pitch down
                control["lon_control"] = -0.3
            elif action == 3:  # Roll right
                control["lat_control"] = 0.3
            elif action == 4:  # Roll left
                control["lat_control"] = -0.3
            elif action == 5:  # Rudder right
                control["rudder_control"] = 0.3
            elif action == 6:  # Rudder left
                control["rudder_control"] = -0.3
            elif action == 7:  # Throttle increase
                control["throttle"] = min(1.0, control["throttle"] + 0.1)
            elif action == 8:  # Throttle decrease
                control["throttle"] = max(0.0, control["throttle"] - 0.1)
            
            return control
    
    def _get_observation(self) -> np.ndarray:
        """Get current environment observation state
        
        Returns:
            Observation state array
        """
        # Get attitude data
        attitude = self.client.get_attitude()
        
        # Get angular rates
        p = self.client.get_dref("sim/flightmodel/position/P")
        q = self.client.get_dref("sim/flightmodel/position/Q")
        r = self.client.get_dref("sim/flightmodel/position/R")
        
        # Get altitude
        alt = self.client.get_dref("sim/flightmodel/position/y_agl")
        
        # Get latitude, longitude and elevation
        lat = self.client.get_dref("sim/flightmodel/position/latitude")
        lon = self.client.get_dref("sim/flightmodel/position/longitude")
        elevation = self.client.get_dref("sim/flightmodel/position/elevation")
        
        # If using random target state, calculate deviation from target
        if self.random_desired_state:
            roll_error = self.desired_state["roll"] - attitude["roll"]
            pitch_error = self.desired_state["pitch"] - attitude["pitch"]
        else:
            roll_error = attitude["roll"]
            pitch_error = attitude["pitch"]
        
        # Normalize heading to [-180, 180]
        heading = attitude["heading"]
        heading_norm = ((heading + 180.0) % 360.0) - 180.0

        # Build observation state
        observation = np.array([
            roll_error,  # roll angle deviation
            pitch_error,  # pitch angle deviation
            heading,  # heading (keep raw 0..360 in obs to match space)
            p,  # roll rate
            q,  # pitch rate
            r,  # yaw rate
            attitude["alpha"],  # angle of attack
            attitude["beta"],  # sideslip angle
            alt,  # height above ground
            lat,  # latitude
            lon,  # longitude
            elevation  # altitude
        ], dtype=np.float32)
        
        return observation
    
    def _compute_reward(self, action: Union[int, np.ndarray], observation: np.ndarray) -> Tuple[float, bool]:
        """Calculate reward and whether terminated
        
        Args:
            action: Executed action
            observation: Current observation state
            
        Returns:
            reward: Reward value
            terminated: Whether terminated
        """
        # Check if crashed
        crashed = self.client.check_crash()
        
        # Get angle of attack and altitude
        alpha = observation[6]  # angle of attack
        alt = observation[8]  # height above ground
        
        # Get roll and pitch deviation
        roll_error = abs(observation[0])
        pitch_error = abs(observation[1])
        
        # Termination conditions
        terminated = False
        
        # Terminate if crashed or altitude too low
        if crashed or alt < 200:
            terminated = True
            reward = -10.0  # Severe penalty
            return reward, terminated
        
        # Angle of attack too large may cause stall
        if abs(alpha) > 15:
            terminated = True
            reward = -5.0  # Moderate penalty
            return reward, terminated
        
        # Reward calculation based on attitude deviation
        # Smaller deviation, higher reward
        error_sum = (roll_error + pitch_error) / 180.0  # Normalize to [0,1] range
        reward = 1.0 - min(1.0, error_sum)  # Control within [0,1] range
        
        # Penalty factors
        # Altitude too low (but not terminated) gets penalty
        if alt < 500:
            reward *= 0.5
        
        # Angle of attack too large (but not terminated) gets penalty
        if abs(alpha) > 10:
            reward *= 0.7
        
        return reward, terminated
    
    def _get_position(self) -> Dict[str, float]:
        """Get current position and attitude data
        
        Returns:
            Dictionary containing position and attitude
        """
        position_data = list(self.client.get_position())
        if len(position_data) < 13:
            # Return default values
            return {
                "lat": 0.0, "lon": 0.0, "alt": 0.0, "agl": 0.0,
                "roll": 0.0, "pitch": 0.0, "heading": 0.0,
                "vx": 0.0, "vy": 0.0, "vz": 0.0,
                "p": 0.0, "q": 0.0, "r": 0.0
            }
        
        return {
            "lat": position_data[0],  # latitude
            "lon": position_data[1],  # longitude
            "alt": position_data[2],  # altitude
            "agl": position_data[3],  # height above ground
            "roll": position_data[4],  # roll angle
            "pitch": position_data[5],  # pitch angle
            "heading": position_data[6],  # heading
            "vx": position_data[7],  # east velocity
            "vy": position_data[8],  # up velocity
            "vz": position_data[9],  # south velocity
            "p": position_data[10],  # roll rate
            "q": position_data[11],  # pitch rate
            "r": position_data[12]   # yaw rate
        }
    
    def _get_random_starting_position(self) -> Tuple[List[float], List[float]]:
        """Generate random starting position and attitude
        
        Returns:
            position: [latitude, longitude, altitude]
            orientation: [roll, pitch, heading]
        """
        # Use the configured starting position
        lat = self.starting_latitude
        lon = self.starting_longitude
        alt = self.starting_altitude
        
        # Random attitude
        roll = np.random.uniform(-self.starting_roll_range, self.starting_roll_range)
        pitch = np.random.uniform(-self.starting_pitch_range, self.starting_pitch_range)
        heading = np.random.uniform(0, 360)
        
        return [lat, lon, alt], [roll, pitch, heading]
    
    def _get_random_desired_state(self) -> Dict[str, float]:
        """Generate random target attitude
        
        Returns:
            Target attitude dictionary, containing roll and pitch
        """
        roll = np.random.uniform(-self.desired_roll_range, self.desired_roll_range)
        pitch = np.random.uniform(-self.desired_pitch_range, self.desired_pitch_range)
        
        return {"roll": roll, "pitch": pitch}
    
    def _calculate_velocity(self, pitch: float, heading: float, airspeed: float) -> Dict[str, float]:
        """Calculate velocity components for a given attitude and airspeed
        
        Args:
            pitch: Pitch angle (degrees)
            heading: Heading (degrees)
            airspeed: Airspeed (meters/second)
            
        Returns:
            Dictionary of velocity components
        """
        # Convert to radians
        pitch_rad = math.radians(pitch)
        heading_rad = math.radians(heading)
        
        # Calculate velocity components in each direction
        # Vertical velocity component
        vy = airspeed * math.sin(pitch_rad)
        
        # Horizontal velocity component
        horizontal_speed = airspeed * math.cos(pitch_rad)
        
        # Decompose into east and north components
        vx = horizontal_speed * math.sin(heading_rad)
        vz = -horizontal_speed * math.cos(heading_rad)
        
        # Return velocity DataRefs
        return {
            "sim/flightmodel/position/local_vx": vx,
            "sim/flightmodel/position/local_vy": vy,
            "sim/flightmodel/position/local_vz": vz
        } 
