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

# 尝试导入XPlaneGym环境
try:
    import XPlaneGym
except ImportError:
    raise ImportError("请确保已安装XPlaneGym包")

def make_env(env_id, rank, seed=0):
    """
    创建环境的辅助函数
    """
    def _init():
        # 使用新版Gymnasium API设置随机种子
        env = gym.make(env_id, render_mode=None)
        env = Monitor(env)
        # 在创建环境时设置随机种子，而不是调用env.seed()
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
    使用PPO训练XPlaneGym环境
    
    参数:
        env_id: 环境ID
        total_timesteps: 总训练步数
        save_path: 模型保存路径
        log_path: 日志保存路径
        save_freq: 保存频率（步数）
        learning_rate: 学习率
        n_steps: 每次更新前收集的步数
        batch_size: 批次大小
    """
    # 创建输出目录
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # 创建向量化环境
    env = DummyVecEnv([make_env(env_id, i) for i in range(1)])
    
    # 正则化观察和奖励
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # 设置回调
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // env.num_envs,
        save_path=save_path,
        name_prefix="ppo_model"
    )
    
    # 创建并训练PPO模型
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log=log_path,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size
    )
    
    print(f"开始训练 PPO 模型，总步数: {total_timesteps}")
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="ppo_run"
    )
    
    # 保存最终模型和正则化参数
    model.save(os.path.join(save_path, "ppo_final_model"))
    env.save(os.path.join(save_path, "vec_normalize.pkl"))
    
    print(f"训练完成！用时: {time.time() - start_time:.2f}秒")
    
    return model

if __name__ == "__main__":
    # 训练示例
    train_ppo(
        env_id="XPlane-v0",  # 请确保这是有效的XPlaneGym环境ID
        total_timesteps=100000,  # 为演示目的减少了步数
        save_freq=10000,
        learning_rate=3e-4
    ) 