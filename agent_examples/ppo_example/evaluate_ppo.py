#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 导入XPlaneGym
try:
    import XPlaneGym
except ImportError:
    raise ImportError("请确保已安装XPlaneGym包")

def evaluate_model(
    model_path,
    vec_normalize_path=None,
    env_id="XPlane-v0",
    num_episodes=5,
    deterministic=True,
    render=True
):
    """
    评估训练好的PPO模型
    
    参数:
        model_path: 已保存模型的路径
        vec_normalize_path: 环境正则化参数的路径
        env_id: 环境ID
        num_episodes: 评估的回合数
        deterministic: 是否使用确定性动作
        render: 是否渲染环境
    """
    # 创建环境
    env = gym.make(env_id, render_mode="human" if render else None)
    env = DummyVecEnv([lambda: env])
    
    # 如果有正则化参数，加载它们
    if vec_normalize_path is not None and os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, env)
        # 在评估时不更新正则化参数
        env.training = False
        env.norm_reward = False
    
    # 加载模型
    model = PPO.load(model_path)
    
    # 评估模型
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
                print(f"回合 {i+1} 完成: 奖励 = {episode_reward}, 步数 = {step}")
                episode_rewards.append(episode_reward)
                break
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"评估完成! 平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    
    return mean_reward, std_reward

if __name__ == "__main__":
    # 评估示例
    model_path = "./models/ppo_final_model.zip"  # 确保路径与训练时保存的路径一致
    vec_normalize_path = "./models/vec_normalize.pkl"
    
    evaluate_model(
        model_path=model_path,
        vec_normalize_path=vec_normalize_path,
        env_id="XPlane-v0",
        num_episodes=3,
        deterministic=True,
        render=True
    ) 