"""
Test Discrete Environment Functionality

This script tests the specific functionality and action mapping of the XPlaneDiscreteEnv discrete environment.
"""

import sys
import os
import time

# Add project path to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import XPlaneGym

def test_discrete_env(max_steps=10):
    """Test discrete environment functionality"""
    print("Testing discrete environment: XPlane-Discrete-v0")
    
    # Create environment
    try:
        env = gym.make("XPlane-Discrete-v0")
        print("✓ Discrete environment created successfully")
        print(f"  - Action space: {env.action_space}")
        # Check if it's a discrete space
        if not isinstance(env.action_space, gym.spaces.Discrete):
            print("✗ Action space is not a discrete space!")
            env.close()
            return False
            
        # Get the unwrapped environment (remove wrappers)
        unwrapped_env = env.unwrapped
        print(f"  - Wrapper type: {type(env)}")
        print(f"  - Unwrapped environment type: {type(unwrapped_env)}")
    except Exception as e:
        print(f"✗ Failed to create discrete environment: {e}")
        return False
    
    # Test action mapping
    try:
        # Try to get the action_map attribute from the unwrapped environment
        action_map = getattr(unwrapped_env, "action_map", None)
        if action_map is None:
            print("✗ Could not find action_map attribute in unwrapped environment")
            env.close()
            return False
        else:
            print("✓ Found action_map attribute")
            
        print("Action mapping:")
        for action_idx, action_name in action_map.items():
            print(f"  - Action {action_idx}: {action_name}")
    except Exception as e:
        print(f"✗ Failed to get action mapping: {e}")
        env.close()
        return False
    
    # Reset environment
    try:
        obs, info = env.reset()
        print("✓ Environment reset successful")
    except Exception as e:
        print(f"✗ Environment reset failed: {e}")
        env.close()
        return False
    
    # Test all discrete actions
    print("Testing all discrete actions:")
    try:
        for action in range(env.action_space.n):
            print(f"  - Executing action {action}: {unwrapped_env.action_map[action]}")
            observation, reward, terminated, truncated, info = env.step(action)
            print(f"    Reward: {reward:.4f}")
            if terminated or truncated:
                print(f"    Environment terminated, resetting environment...")
                obs, info = env.reset()
    except Exception as e:
        print(f"✗ Action execution failed: {e}")
        env.close()
        return False
    
    # Execute some random steps
    print(f"\nExecuting {max_steps} random actions:")
    try:
        obs, info = env.reset()
        for i in range(max_steps):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            print(f"  - Step {i+1}: action={action} ({unwrapped_env.action_map[action]}), reward={reward:.4f}")
            
            if terminated or truncated:
                print(f"    Environment terminated, reason: {'termination condition' if terminated else 'max steps'}")
                break
    except Exception as e:
        print(f"✗ Random action execution failed: {e}")
        env.close()
        return False
    
    # Close environment
    env.close()
    print("All discrete environment tests passed!")
    return True

if __name__ == "__main__":
    # Get number of steps from command line arguments (if provided)
    max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    
    test_discrete_env(max_steps) 