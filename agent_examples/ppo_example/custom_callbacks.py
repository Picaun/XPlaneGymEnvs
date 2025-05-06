#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

class TensorboardCallback(BaseCallback):
    """
    自定义回调，用于记录额外的值到Tensorboard
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        # 累计当前回合奖励和长度
        self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1
        
        # 检查回合是否结束
        if self.locals["dones"][0]:
            # 记录回合奖励和长度
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # 重置计数器
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            # 计算最近10个回合的平均奖励
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                self.logger.record("rollout/mean_reward_last_10", mean_reward)
                
                # 如果有至少10个回合，记录奖励方差
                if len(self.episode_rewards) >= 10:
                    reward_variance = np.var(self.episode_rewards[-10:])
                    self.logger.record("rollout/reward_variance", reward_variance)
        
        return True

class RewardVisualizer(BaseCallback):
    """
    回调函数，用于可视化训练过程中的奖励
    """
    def __init__(self, log_dir, verbose=0, plot_freq=10000):
        super(RewardVisualizer, self).__init__(verbose)
        self.log_dir = log_dir
        self.plot_freq = plot_freq
        self.rewards = []
        self.timesteps = []
        self.current_episode_reward = 0
        
    def _on_step(self) -> bool:
        # 累计当前回合奖励
        self.current_episode_reward += self.locals["rewards"][0]
        
        # 检查回合是否结束
        if self.locals["dones"][0]:
            # 保存回合奖励和对应的总步数
            self.rewards.append(self.current_episode_reward)
            self.timesteps.append(self.num_timesteps)
            self.current_episode_reward = 0
            
            # 每plot_freq步绘制一次奖励图
            if self.num_timesteps % self.plot_freq == 0 and len(self.rewards) > 0:
                self._plot_rewards()
        
        return True
    
    def _plot_rewards(self):
        """绘制训练奖励图"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.timesteps, self.rewards, 'b-')
        plt.xlabel('训练步数')
        plt.ylabel('回合奖励')
        plt.title('训练过程中的回合奖励')
        plt.grid(True)
        
        # 添加移动平均线
        if len(self.rewards) >= 10:
            moving_avg = np.convolve(self.rewards, np.ones(10)/10, mode='valid')
            plt.plot(self.timesteps[9:], moving_avg, 'r-', label='10回合移动平均')
            plt.legend()
        
        # 保存图像
        os.makedirs(self.log_dir, exist_ok=True)
        plt.savefig(os.path.join(self.log_dir, f'rewards_{self.num_timesteps}.png'))
        plt.close()

class CombinedCallback(BaseCallback):
    """
    组合多个回调函数
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
            # 将当前locals和globals传递给每个回调
            callback.locals = self.locals
            callback.globals = self.globals
            
            # 调用回调的_on_step方法
            continue_training = callback._on_step() and continue_training
            
        return continue_training
    
    def _on_training_end(self):
        for callback in self.callbacks:
            callback._on_training_end()

# 使用示例
def get_callbacks(save_path, log_path, save_freq=10000):
    """
    创建组合回调
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