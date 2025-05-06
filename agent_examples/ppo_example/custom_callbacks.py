#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional values to Tensorboard
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        # Accumulate current episode reward and length
        self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1
        
        # Check if episode has ended
        if self.locals["dones"][0]:
            # Log episode reward and length
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Reset counters
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            # Calculate average reward of last 10 episodes
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                self.logger.record("rollout/mean_reward_last_10", mean_reward)
                
                # If we have at least 10 episodes, log reward variance
                if len(self.episode_rewards) >= 10:
                    reward_variance = np.var(self.episode_rewards[-10:])
                    self.logger.record("rollout/reward_variance", reward_variance)
        
        return True

class RewardVisualizer(BaseCallback):
    """
    Callback for visualizing rewards during training
    """
    def __init__(self, log_dir, verbose=0, plot_freq=10000):
        super(RewardVisualizer, self).__init__(verbose)
        self.log_dir = log_dir
        self.plot_freq = plot_freq
        self.rewards = []
        self.timesteps = []
        self.current_episode_reward = 0
        
    def _on_step(self) -> bool:
        # Accumulate current episode reward
        self.current_episode_reward += self.locals["rewards"][0]
        
        # Check if episode has ended
        if self.locals["dones"][0]:
            # Save episode reward and corresponding total steps
            self.rewards.append(self.current_episode_reward)
            self.timesteps.append(self.num_timesteps)
            self.current_episode_reward = 0
            
            # Plot rewards every plot_freq steps
            if self.num_timesteps % self.plot_freq == 0 and len(self.rewards) > 0:
                self._plot_rewards()
        
        return True
    
    def _plot_rewards(self):
        """Plot training rewards"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.timesteps, self.rewards, 'b-')
        plt.xlabel('Training Steps')
        plt.ylabel('Episode Reward')
        plt.title('Episode Rewards During Training')
        plt.grid(True)
        
        # Add moving average line
        if len(self.rewards) >= 10:
            moving_avg = np.convolve(self.rewards, np.ones(10)/10, mode='valid')
            plt.plot(self.timesteps[9:], moving_avg, 'r-', label='10-episode moving average')
            plt.legend()
        
        # Save the image
        os.makedirs(self.log_dir, exist_ok=True)
        plt.savefig(os.path.join(self.log_dir, f'rewards_{self.num_timesteps}.png'))
        plt.close()

class CombinedCallback(BaseCallback):
    """
    Combine multiple callbacks
    """
    def __init__(self, callbacks, verbose=0):
        super(CombinedCallback, self).__init__(verbose)
        self.callbacks = callbacks
    
    def _on_training_start(self):
        for callback in self.callbacks:
            callback._on_training_start()
    
    def _on_step(self) -> bool:
        continue_training = True
        
        for callback in self.callbacks:
            # Pass current locals and globals to each callback
            callback.locals = self.locals
            callback.globals = self.globals
            
            # Call _on_step method of the callback
            continue_training = callback._on_step() and continue_training
            
        return continue_training
    
    def _on_training_end(self):
        for callback in self.callbacks:
            callback._on_training_end()

# Usage example
def get_callbacks(save_path, log_path, save_freq=10000):
    """
    Create combined callbacks
    """
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_path,
        name_prefix="ppo_model"
    )
    
    tensorboard_callback = TensorboardCallback()
    
    reward_visualizer = RewardVisualizer(
        log_dir=os.path.join(log_path, "reward_plots"),
        plot_freq=save_freq
    )
    
    return CombinedCallback([
        checkpoint_callback,
        tensorboard_callback,
        reward_visualizer
    ]) 