#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import XPlaneGym
try:
    import XPlaneGym
except ImportError:
    raise ImportError("Please make sure XPlaneGym package is installed")

def run_trained_agent(
    model_path,
    vec_normalize_path=None,
    env_id="XPlane-v0",
    num_episodes=1,
    deterministic=True
):
    """
    Run agent using a trained model
    
    Parameters:
        model_path: Path to the saved model
        vec_normalize_path: Path to environment normalization parameters
        env_id: Environment ID
        num_episodes: Number of episodes to run
        deterministic: Whether to use deterministic actions
    """
    # Create environment
    env = gym.make(env_id, render_mode="human")
    env = DummyVecEnv([lambda: env])
    
    # If normalization parameters exist, load them
    if vec_normalize_path is not None and os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, env)
        # Do not update normalization parameters during evaluation
        env.training = False
        env.norm_reward = False
    
    # Load model
    model = PPO.load(model_path)
    
    print(f"Model loaded, starting to run in the environment...")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        print(f"Starting episode {episode+1}/{num_episodes}")
        
        while not done:
            # Predict action using the model
            action, _states = model.predict(obs, deterministic=deterministic)
            
            # Execute action
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            reward = rewards[0]
            total_reward += reward
            step += 1
            
            if step % 100 == 0:
                print(f"Steps: {step}, Cumulative reward: {total_reward:.2f}")
        
        print(f"Episode {episode+1} completed! Total steps: {step}, Total reward: {total_reward:.2f}")
    
    print("All episodes completed!")
    
    return total_reward

if __name__ == "__main__":
    # Usage example
    model_path = "./models/ppo_final_model.zip"
    vec_normalize_path = "./models/vec_normalize.pkl"
    
    run_trained_agent(
        model_path=model_path,
        vec_normalize_path=vec_normalize_path,
        env_id="XPlane-v0",
        num_episodes=1,
        deterministic=True
    ) 