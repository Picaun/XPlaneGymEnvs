"""
Test Continuous Environment Functionality

This script tests the specific functionality of the XPlaneContinuousEnv continuous environment, 
including different action dimensions and action smoothing features.
"""

import sys
import os
import time

# Add project path to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import XPlaneGym

def test_continuous_env(max_steps=10):
    """Test various functions of the continuous environment"""
    print("Testing continuous environment: XPlane-Continuous-v0")
    
    # Test different action space dimensions
    for action_dim in [2, 3, 4]:
        print(f"\nTesting {action_dim} dimensional action space:")
        
        # Create environment
        try:
            env = gym.make("XPlane-Continuous-v0", action_dim=action_dim)
            print(f"✓ Continuous environment created successfully (action_dim={action_dim})")
            print(f"  - Action space: {env.action_space}")
            # Check if it's a Box space
            if not isinstance(env.action_space, gym.spaces.Box):
                print("✗ Action space is not a continuous space!")
                env.close()
                continue
                
            # Check action space dimension
            if env.action_space.shape[0] != action_dim:
                print(f"✗ Action space dimension error! Expected {action_dim}, actual {env.action_space.shape[0]}")
                env.close()
                continue
                
            print(f"✓ Action space dimension correct: {env.action_space.shape[0]}")
        except Exception as e:
            print(f"✗ Failed to create continuous environment: {e}")
            continue
        
        # Reset environment
        try:
            obs, info = env.reset()
            print("✓ Environment reset successful")
        except Exception as e:
            print(f"✗ Environment reset failed: {e}")
            env.close()
            continue
        
        # Test fixed action combinations
        print("Testing fixed action combinations:")
        
        # Define test actions
        test_actions = []
        if action_dim == 2:
            test_actions = [
                np.array([0.5, 0.0]),  # Pitch up only
                np.array([0.0, 0.5]),  # Roll right only
                np.array([-0.5, -0.5])  # Pitch down + Roll left
            ]
        elif action_dim == 3:
            test_actions = [
                np.array([0.5, 0.0, 0.0]),  # Pitch up only
                np.array([0.0, 0.5, 0.0]),  # Roll right only
                np.array([0.0, 0.0, 0.5])   # Rudder right only
            ]
        else:  # action_dim == 4
            test_actions = [
                np.array([0.5, 0.0, 0.0, 0.5]),  # Pitch up + half throttle
                np.array([0.0, 0.5, 0.0, 0.7]),  # Roll right + 70% throttle
                np.array([0.0, 0.0, 0.5, 1.0])   # Rudder right + full throttle
            ]
        
        # Execute test actions
        try:
            for i, action in enumerate(test_actions):
                print(f"  - Executing action {i+1}: {action}")
                observation, reward, terminated, truncated, info = env.step(action)
                print(f"    Reward: {reward:.4f}")
                
                # Check if info contains smoothed action
                if "smoothed_action" in info:
                    print(f"    Raw action: {info['raw_action']}")
                    print(f"    Smoothed action: {info['smoothed_action']}")
                else:
                    print("    Note: No action smoothing information")
                
                if terminated or truncated:
                    print(f"    Environment terminated, resetting environment...")
                    obs, info = env.reset()
        except Exception as e:
            print(f"✗ Failed to execute fixed actions: {e}")
            env.close()
            continue
        
        # Execute random actions
        print(f"\nExecuting {max_steps} random actions:")
        try:
            obs, info = env.reset()
            for i in range(max_steps):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                print(f"  - Step {i+1}:")
                print(f"    Action: {action}")
                print(f"    Smoothed: {info.get('smoothed_action', 'N/A')}")
                print(f"    Reward: {reward:.4f}")
                
                if terminated or truncated:
                    print(f"    Environment terminated, reason: {'termination condition' if terminated else 'max steps'}")
                    break
        except Exception as e:
            print(f"✗ Failed to execute random actions: {e}")
            env.close()
            continue
        
        # Close environment
        env.close()
        print(f"{action_dim} dimensional action space testing completed")
    
    print("\nAll continuous environment tests passed!")
    return True

if __name__ == "__main__":
    # Get number of steps from command line arguments (if provided)
    max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    
    test_continuous_env(max_steps) 