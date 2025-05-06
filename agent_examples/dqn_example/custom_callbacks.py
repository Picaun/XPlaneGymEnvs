#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

class RewardTracker(BaseCallback):
    """
    Track rewards for each episode
    """
    def __init__(self, verbose=0):
        super(RewardTracker, self).__init__(verbose)
        self.rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Accumulate current episode reward
        self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1
        
        # Check if episode is done
        if self.locals["dones"][0]:
            # Store episode reward
            self.rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Log to tensorboard
            self.logger.record("rollout/episode_reward", self.current_episode_reward)
            self.logger.record("rollout/episode_length", self.current_episode_length)
            
            # Track average reward
            if len(self.rewards) > 0:
                avg_reward = np.mean(self.rewards[-10:] if len(self.rewards) >= 10 else self.rewards)
                self.logger.record("rollout/avg_reward_last_10", avg_reward)
            
            # Reset
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
        return True

class ExplorationTracker(BaseCallback):
    """
    Track changes in DQN exploration rate
    """
    def __init__(self, verbose=0):
        super(ExplorationTracker, self).__init__(verbose)
        
    def _on_step(self) -> bool:
        # Record current exploration rate
        if hasattr(self.model, "exploration_rate"):
            self.logger.record("rollout/exploration_rate", self.model.exploration_rate)
        return True

class QValueTracker(BaseCallback):
    """
    Track changes in DQN Q-values
    """
    def __init__(self, verbose=0, log_freq=1000):
        super(QValueTracker, self).__init__(verbose)
        self.log_freq = log_freq
        self.q_values = []
        
    def _on_step(self) -> bool:
        # Record Q-values every log_freq steps
        if self.num_timesteps % self.log_freq == 0:
            # Get Q-values for current observation
            obs = self.locals["new_obs"]
            # Calculate Q-values for all actions
            q_values = self.model.q_net(self.model.q_net.obs_to_tensor(obs)[0])[0].detach().cpu().numpy()
            
            # Record max Q-value
            max_q = float(np.max(q_values))
            min_q = float(np.min(q_values))
            mean_q = float(np.mean(q_values))
            
            self.q_values.append(max_q)
            self.logger.record("rollout/max_q_value", max_q)
            self.logger.record("rollout/min_q_value", min_q)
            self.logger.record("rollout/mean_q_value", mean_q)
            
        return True

class DQNVisualizer(BaseCallback):
    """
    Visualize DQN training progress
    """
    def __init__(self, log_dir, verbose=0, plot_freq=10000):
        super(DQNVisualizer, self).__init__(verbose)
        self.log_dir = log_dir
        self.plot_freq = plot_freq
        
        # Store data
        self.timesteps = []
        self.rewards = []
        self.explorations = []
        self.current_episode_reward = 0
        
    def _init_callback(self):
        os.makedirs(self.log_dir, exist_ok=True)
        
    def _on_step(self) -> bool:
        # Accumulate current episode reward
        self.current_episode_reward += self.locals["rewards"][0]
        
        # If episode is done
        if self.locals["dones"][0]:
            self.timesteps.append(self.num_timesteps)
            self.rewards.append(self.current_episode_reward)
            
            if hasattr(self.model, "exploration_rate"):
                self.explorations.append(self.model.exploration_rate)
            
            # Reset episode reward
            self.current_episode_reward = 0
            
            # Plot every plot_freq steps
            if self.num_timesteps % self.plot_freq == 0 and len(self.rewards) > 0:
                self._plot_training_progress()
                
        return True
    
    def _plot_training_progress(self):
        """Plot training progress charts"""
        # Create chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
        
        # Plot reward chart
        ax1.plot(self.timesteps, self.rewards, 'b-')
        ax1.set_ylabel('Episode Reward')
        ax1.set_title('DQN Training Progress')
        ax1.grid(True)
        
        # Add moving average
        if len(self.rewards) >= 10:
            window_size = 10
            moving_avg = np.convolve(self.rewards, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(self.timesteps[window_size-1:], moving_avg, 'r-', label='10-episode moving average')
            ax1.legend()
        
        # Plot exploration rate chart
        if len(self.explorations) > 0:
            ax2.plot(self.timesteps, self.explorations, 'g-')
            ax2.set_xlabel('Training Steps')
            ax2.set_ylabel('Exploration Rate')
            ax2.grid(True)
        
        # Save chart
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'training_progress_{self.num_timesteps}.png'))
        plt.close(fig)

def get_callbacks(save_path, log_path, save_freq=10000):
    """
    Create callback combination for DQN training
    """
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_path,
        name_prefix="dqn_model"
    )
    
    reward_tracker = RewardTracker()
    exploration_tracker = ExplorationTracker()
    q_value_tracker = QValueTracker(log_freq=1000)
    visualizer = DQNVisualizer(
        log_dir=os.path.join(log_path, "visualizations"),
        plot_freq=save_freq
    )
    
    # Return callback list
    return [
        checkpoint_callback,
        reward_tracker,
        exploration_tracker,
        q_value_tracker,
        visualizer
    ] 