#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Try to import XPlaneGym environment
try:
    import XPlaneGym
except ImportError:
    raise ImportError("Please make sure XPlaneGym package is installed")

def make_env(env_id, rank, seed=0):
    """
    Helper function to create environments
    """
    def _init():
        # Use the new Gymnasium API to set random seed
        env = gym.make(env_id, render_mode=None)
        env = Monitor(env)
        # Set random seed when creating environment, instead of calling env.seed()
        return env
    return _init

def train_ppo(
    env_id="XPlane-v0",
    total_timesteps=1000000,
    save_path="./models",
    log_path="./logs",
    save_freq=10000,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64
):
    """
    Train XPlaneGym environment using PPO
    
    Parameters:
        env_id: Environment ID
        total_timesteps: Total training steps
        save_path: Model save path
        log_path: Log save path
        save_freq: Save frequency (steps)
        learning_rate: Learning rate
        n_steps: Number of steps to collect before each update
        batch_size: Batch size
    """
    # Create output directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # Create vectorized environment
    env = DummyVecEnv([make_env(env_id, i) for i in range(1)])
    
    # Normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // env.num_envs,
        save_path=save_path,
        name_prefix="ppo_model"
    )
    
    # Create and train PPO model
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log=log_path,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size
    )
    
    print(f"Starting PPO model training, total steps: {total_timesteps}")
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="ppo_run"
    )
    
    # Save final model and normalization parameters
    model.save(os.path.join(save_path, "ppo_final_model"))
    env.save(os.path.join(save_path, "vec_normalize.pkl"))
    
    print(f"Training completed! Time taken: {time.time() - start_time:.2f} seconds")
    
    return model

if __name__ == "__main__":
    # Training example
    train_ppo(
        env_id="XPlane-v0",  # Please ensure this is a valid XPlaneGym environment ID
        total_timesteps=100000,  # Reduced steps for demonstration purposes
        save_freq=10000,
        learning_rate=3e-4
    ) 