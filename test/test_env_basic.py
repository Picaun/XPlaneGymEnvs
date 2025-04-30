"""
Test Basic Environment Functionality

This script tests the creation, reset, action execution, and rendering functionality of the XPlaneEnv basic environment.
"""

import sys
import os
import time

# Add project path to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import XPlaneGym

def test_env_basic(env_id="XPlane-v0", max_steps=10):
    """Test basic environment functionality"""
    print(f"Testing environment: {env_id}")
    
    # Create environment
    try:
        env = gym.make(env_id)
        print(f"✓ Environment created successfully")
        print(f"  - Observation space: {env.observation_space}")
        print(f"  - Action space: {env.action_space}")
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return False
    
    # Reset environment
    try:
        obs, info = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  - Observation shape: {obs.shape}")
        print(f"  - Initial state: Roll deviation={obs[0]:.2f}°, Pitch deviation={obs[1]:.2f}°")
        print(f"  - Position: Latitude={obs[9]:.6f}°, Longitude={obs[10]:.6f}°, Altitude={obs[11]:.2f} meters")
    except Exception as e:
        print(f"✗ Environment reset failed: {e}")
        env.close()
        return False
    
    # Execute some random actions
    print(f"Executing {max_steps} random actions:")
    try:
        for i in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"  - Step {i+1}:")
            print(f"    Action: {action}")
            print(f"    Reward: {reward:.4f}")
            print(f"    Attitude: Roll deviation={obs[0]:.2f}°, Pitch deviation={obs[1]:.2f}°")
            print(f"    Position: Latitude={obs[9]:.6f}°, Longitude={obs[10]:.6f}°, Altitude={obs[11]:.2f} meters")
            
            if terminated or truncated:
                print(f"    Environment terminated, reason: {'termination condition' if terminated else 'max steps'}")
                break
    except Exception as e:
        print(f"✗ Action execution failed: {e}")
        env.close()
        return False
    
    # Close environment
    try:
        env.close()
        print("✓ Environment closed successfully")
    except Exception as e:
        print(f"✗ Environment closure failed: {e}")
        return False
    
    print("All basic environment tests passed!")
    return True

if __name__ == "__main__":
    # Get environment ID from command line arguments (if provided)
    env_id = sys.argv[1] if len(sys.argv) > 1 else "XPlane-v0"
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    test_env_basic(env_id, max_steps) 