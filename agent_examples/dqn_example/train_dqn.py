#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Import for custom callbacks
from custom_callbacks import get_callbacks

# Try to import XPlaneGym environment
try:
    import XPlaneGym
except ImportError:
    raise ImportError("Please ensure XPlaneGym package is installed")

def make_env(env_id, rank=0):
    """
    Helper function to create environment
    """
    def _init():
        env = gym.make(env_id, render_mode=None)
        env = Monitor(env)
        return env
    return _init

def train_dqn(
    env_id="XPlane-v0",
    total_timesteps=500000,
    save_path="./models",
    log_path="./logs",
    save_freq=10000,
    learning_rate=1e-4,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    tau=1.0,
    target_update_interval=1000,
    train_freq=4,
    gradient_steps=1,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    use_custom_callbacks=True
):
    """
    Train XPlaneGym environment using DQN
    
    Parameters:
        env_id: Environment ID
        total_timesteps: Total training steps
        save_path: Model save path
        log_path: Log save path
        save_freq: Save frequency (steps)
        learning_rate: Learning rate
        buffer_size: Experience replay buffer size
        learning_starts: Steps before learning starts
        batch_size: Batch size
        gamma: Discount factor
        tau: Target network soft update coefficient
        target_update_interval: Target network update interval
        train_freq: Training frequency
        gradient_steps: Gradient steps per update
        exploration_fraction: Exploration steps as fraction of total steps
        exploration_initial_eps: Initial exploration rate
        exploration_final_eps: Final exploration rate
        use_custom_callbacks: Whether to use custom callbacks
    """
    # Create output directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # Create environment
    env = DummyVecEnv([make_env(env_id)])
    
    # Set up callbacks
    if use_custom_callbacks:
        # Use custom callback combination
        callbacks = get_callbacks(save_path, log_path, save_freq)
    else:
        # Only use model save callback
        callbacks = CheckpointCallback(
            save_freq=save_freq,
            save_path=save_path,
            name_prefix="dqn_model"
        )
    
    # Create and train DQN model
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log=log_path,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        target_update_interval=target_update_interval,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps
    )
    
    print(f"Starting DQN model training, total steps: {total_timesteps}")
    print(f"Exploration settings: initial rate={exploration_initial_eps}, final rate={exploration_final_eps}, fraction={exploration_fraction}")
    print(f"Optimization settings: learning rate={learning_rate}, batch size={batch_size}, buffer size={buffer_size}")
    
    start_time = time.time()
    
    # Train model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name="dqn_run"
    )
    
    # Save final model
    model.save(os.path.join(save_path, "dqn_final_model"))
    
    print(f"Training complete! Time taken: {time.time() - start_time:.2f} seconds")
    
    # Evaluate final model performance
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Final model evaluation: Average reward = {mean_reward:.2f} ± {std_reward:.2f}")
    
    return model

if __name__ == "__main__":
    # Training example
    train_dqn(
        env_id="XPlane-v0",  # Please ensure this is a valid XPlaneGym environment ID
        total_timesteps=50000,  # Reduced steps for demonstration
        save_freq=5000,
        learning_rate=1e-4,
        buffer_size=10000,
        exploration_fraction=0.2,
        use_custom_callbacks=True  # Use custom callbacks
    ) 