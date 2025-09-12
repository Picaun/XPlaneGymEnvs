# XPlane Gym: A Reinforcement Learning Environment Compatible with X-Plane
[中文](https://github.com/Picaun/XPlaneGymEnvs/blob/main/README_zh.md) | [English](https://github.com/Picaun/XPlaneGymEnvs/blob/main/README.md)
XPlaneGymEnvs is an X-Plane flight simulator environment compliant with the OpenAI Gym interface, specifically designed for reinforcement learning research. It provides seamless integration with the X-Plane simulator, supports both discrete and continuous action spaces, and can be used to train agents to perform flight control tasks.

## Installation Requirements

- X-Plane 12 Flight Simulator（Perhaps it will also work in lower versions）
- Python 3.8+
- gymnasium
- numpy

## Installation

```
# Clone the repository
git clone https://github.com/Picaun/XPlaneGymEnvs.git
cd XPlaneGymEnvs

# Install the project and its dependencies
pip install -e .
```

## Available Environments

- `XPlane-v0`: Basic environment, configurable as discrete or continuous action space
- `XPlane-custom-v0`: custom environment, Import XPlaneEnv class to personalize the environment

## Quick Start

### 1. Installation

```
pip install -e .
```

### 2. Launch X-Plane

1. Start the X-Plane flight simulator
2. Ensure that the UDP communication port is set to 49000 (default value) in "Settings > Network"
3. Starting a new flight, you can set the weather conditions and flight duration by yourself
   (note that this function is not yet directly controlled with X-Plane 12 in the XPlaneGymEnvs interface)

### 3. use agent_examples
```
cd agent_examples/dqn_example
```
```
python train_dqn.py
```


## Environment Parameter Configuration

Various parameters can be configured when creating the environment:

```python
env = gym.make(
    "XPlane-Continuous-v0",
    ip='127.0.0.1',                # X-Plane IP address
    port=49000,                    # X-Plane UDP port
    timeout=1.0,                   # Communication timeout
    pause_delay=0.05,              # Action execution pause delay
    starting_latitude=37.558,      # Initial latitude (default near Seoul Gimpo International Airport)
    starting_longitude=126.790,    # Initial longitude (default near Seoul Gimpo International Airport)
    starting_altitude=3000.0,      # Initial altitude
    starting_velocity=100.0,       # Initial velocity
    starting_pitch_range=10.0,     # Initial pitch angle random range
    starting_roll_range=20.0,      # Initial roll angle random range
    random_desired_state=True,     # Whether to use random target attitude
    desired_pitch_range=5.0,       # Target pitch angle random range
    desired_roll_range=10.0,       # Target roll angle random range
    action_dim=2                   # Continuous action space dimension
)
```

## Environment Interface

### Observation Space

The default observation space contains 12 continuous values:
1. Roll angle deviation (degrees)
2. Pitch angle deviation (degrees)
3. Heading (degrees)
4. Roll rate (radians/second)
5. Pitch rate (radians/second)
6. Yaw rate (radians/second)
7. Angle of attack (degrees)
8. Sideslip angle (degrees)
9. Height above ground (meters)
10. Latitude (degrees)
11. Longitude (degrees)
12. Altitude (meters)

### Discrete Action Space

The default provides 9 discrete actions:

0. No action
1. Pitch up
2. Pitch down
3. Roll left
4. Roll right
5. Pitch up + Roll left
6. Pitch up + Roll right
7. Pitch down + Roll left
8. Pitch down + Roll right

### Continuous Action Space

Provides 2-4 dimensional continuous actions based on configuration:
- 2D: [Pitch control, Roll control]
- 3D: [Pitch control, Roll control, Rudder control]
- 4D: [Pitch control, Roll control, Rudder control, Throttle]

Each dimension has a range of [-1.0, 1.0], except for throttle which has a range of [0.0, 1.0].

### Reward Function

The reward function is primarily calculated based on the deviation between the current attitude and the target attitude, including:
- Reward for attitude close to target state
- Penalty for excessive attitude deviation
- Penalty for low altitude
- Penalty for excessive angle of attack
- Negative reward for aircraft crash
- Control smoothness reward (in continuous environment)


## Custom Extensions

You can extend the environment functionality by inheriting from the base class:
- Create new environment classes by inheriting from `XPlaneEnv`
- Override the `_compute_reward` method to customize the reward function
- Override the `_get_observation` method to customize the observation space

## Usage Example

```python
import gymnasium as gym
import XPlaneGym
import numpy as np

# Create environment
env = gym.make("XPlane-v0")

# Reset environment to get initial observation
observation, info = env.reset()

# Run 10 episodes
for episode in range(10):
    observation, info = env.reset()
    episode_reward = 0
    step_count = 0
    done = False
    
    print(f"Starting Episode {episode+1}")
    
    # Single episode loop
    while not done and step_count < 100:
        # Sample random action (should use policy network in actual application)
        action = env.action_space.sample()
        
        # Execute action
        observation, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        step_count += 1
        done = terminated or truncated
        
        # Print current state
        print(f"Step {step_count}: Reward = {reward:.2f}, Cumulative Reward = {episode_reward:.2f}")
        print(f"Pitch Deviation: {observation[1]:.2f}°, Roll Deviation: {observation[0]:.2f}°")
        print(f"Position: Latitude {observation[9]:.6f}°, Longitude {observation[10]:.6f}°, Altitude {observation[11]:.2f}m")
        
    print(f"Episode {episode+1} End: Total Steps = {step_count}, Total Reward = {episode_reward:.2f}")
    print("-" * 50)

# Close the environment
env.close()
```
## Issue

Due to limited personal abilities, various problems are inevitable during operation. Fortunately, we can communicate and solve problems in the issue

## License

MIT

## Acknowledgements

This project is based on the following open source projects:
- [XPlaneConnectX](https://github.com/sisl/XPlaneConnectX)
- [GYM_XPLANE_ML](https://github.com/adderbyte/GYM_XPLANE_ML)
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) 
