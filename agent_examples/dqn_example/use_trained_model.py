#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# Import XPlaneGym
try:
    import XPlaneGym
except ImportError:
    raise ImportError("Please ensure XPlaneGym package is installed")

def run_trained_agent(
    model_path,
    env_id="XPlane-v0",
    num_episodes=1,
    deterministic=True,
    render_delay=0.01  # Can add rendering delay for easier observation
):
    """
    Run trained DQN model agent
    
    Parameters:
        model_path: Path to saved model
        env_id: Environment ID
        num_episodes: Number of episodes to run
        deterministic: Whether to use deterministic actions
        render_delay: Delay between steps (seconds)
    """
    # Create environment
    env = gym.make(env_id, render_mode="human")
    env = DummyVecEnv([lambda: env])
    
    # Load model
    model = DQN.load(model_path)
    
    print(f"Model loaded, starting to run in environment...")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        print(f"Starting episode {episode+1}/{num_episodes}")
        
        while not done:
            # Use model to predict action
            action, _states = model.predict(obs, deterministic=deterministic)
            
            # Execute action
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            reward = rewards[0]
            total_reward += reward
            step += 1
            
            # Add optional delay for observation
            if render_delay > 0:
                time.sleep(render_delay)
            
            if step % 100 == 0:
                print(f"Steps: {step}, Cumulative reward: {total_reward:.2f}")
        
        print(f"Episode {episode+1} complete! Total steps: {step}, Total reward: {total_reward:.2f}")
    
    print("All episodes completed!")
    
    return total_reward

if __name__ == "__main__":
    # Usage example
    model_path = "./models/dqn_final_model.zip"
    
    run_trained_agent(
        model_path=model_path,
        env_id="XPlane-v0",
        num_episodes=1,
        deterministic=True
    ) 