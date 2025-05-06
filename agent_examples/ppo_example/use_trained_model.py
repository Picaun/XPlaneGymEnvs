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

def run_trained_agent(
    model_path,
    vec_normalize_path=None,
    env_id="XPlane-v0",
    num_episodes=1,
    deterministic=True
):
    """
    使用训练好的模型运行智能体
    
    参数:
        model_path: 已保存模型的路径
        vec_normalize_path: 环境正则化参数的路径
        env_id: 环境ID
        num_episodes: 运行的回合数
        deterministic: 是否使用确定性动作
    """
    # 创建环境
    env = gym.make(env_id, render_mode="human")
    env = DummyVecEnv([lambda: env])
    
    # 如果有正则化参数，加载它们
    if vec_normalize_path is not None and os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, env)
        # 在评估时不更新正则化参数
        env.training = False
        env.norm_reward = False
    
    # 加载模型
    model = PPO.load(model_path)
    
    print(f"模型已加载，开始在环境中运行...")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        print(f"开始回合 {episode+1}/{num_episodes}")
        
        while not done:
            # 使用模型预测动作
            action, _states = model.predict(obs, deterministic=deterministic)
            
            # 执行动作
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            reward = rewards[0]
            total_reward += reward
            step += 1
            
            if step % 100 == 0:
                print(f"步数: {step}, 累计奖励: {total_reward:.2f}")
        
        print(f"回合 {episode+1} 完成! 总步数: {step}, 总奖励: {total_reward:.2f}")
    
    print("所有回合已完成!")
    
    return total_reward

if __name__ == "__main__":
    # 使用示例
    model_path = "./models/ppo_final_model.zip"
    vec_normalize_path = "./models/vec_normalize.pkl"
    
    run_trained_agent(
        model_path=model_path,
        vec_normalize_path=vec_normalize_path,
        env_id="XPlane-v0",
        num_episodes=1,
        deterministic=True
    ) 