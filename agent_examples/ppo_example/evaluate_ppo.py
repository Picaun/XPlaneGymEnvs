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

def evaluate_model(
    model_path,
    vec_normalize_path=None,
    env_id="XPlane-v0",
    num_episodes=5,
    deterministic=True,
    render=True
):
    """
    Evaluate a trained PPO model
    
    Parameters:
        model_path: Path to the saved model
        vec_normalize_path: Path to environment normalization parameters
        env_id: Environment ID
        num_episodes: Number of episodes for evaluation
        deterministic: Whether to use deterministic actions
        render: Whether to render the environment
    """
    # Create environment
    env = gym.make(env_id, render_mode="human" if render else None)
    env = DummyVecEnv([lambda: env])
    
    # If normalization parameters exist, load them
    if vec_normalize_path is not None and os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, env)
        # Do not update normalization parameters during evaluation
        env.training = False
        env.norm_reward = False
    
    # Load model
    model = PPO.load(model_path)
    
    # Evaluate model
    episode_rewards = []
    for i in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward[0]
            step += 1
            
            if done:
                print(f"Episode {i+1} completed: Reward = {episode_reward}, Steps = {step}")
                episode_rewards.append(episode_reward)
                break
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"Evaluation completed! Average reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    return mean_reward, std_reward

if __name__ == "__main__":
    # Evaluation example
    model_path = "./models/ppo_final_model.zip"  # Make sure the path matches where your model was saved
    vec_normalize_path = "./models/vec_normalize.pkl"
    
    evaluate_model(
        model_path=model_path,
        vec_normalize_path=vec_normalize_path,
        env_id="XPlane-v0",
        num_episodes=3,
        deterministic=True,
        render=True
    ) 