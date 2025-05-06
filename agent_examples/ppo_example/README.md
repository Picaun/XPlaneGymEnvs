# PPO算法训练XPlaneGym示例

这个示例展示了如何使用Stable-Baselines3中的PPO（近端策略优化）算法来训练XPlaneGym环境中的强化学习智能体。

## 环境要求

首先安装必要的依赖：

```bash
pip install -r requirements.txt
```

确保已安装好XPlaneGym包，并且X-Plane模拟器已正确配置。

> **注意**: 本示例使用的是Gymnasium（gym的后继版本）API，请确保使用的是Gymnasium >= 0.28.0。Gymnasium的API与旧版gym有一些区别，特别是在环境重置和步进方法上。

## 文件说明

- `train_ppo.py`: PPO训练主脚本
- `evaluate_ppo.py`: 评估已训练模型的脚本
- `custom_callbacks.py`: 自定义训练回调函数
- `use_trained_model.py`: 使用训练好的模型运行环境
- `requirements.txt`: 项目依赖

## 使用方法

### 训练模型

```bash
python train_ppo.py
```

训练参数可以在脚本中调整，包括：
- 环境ID
- 训练步数
- 学习率
- 批次大小
- 保存频率等

训练过程中，模型会定期保存在`./models`目录，训练日志保存在`./logs`目录。

### 评估模型

训练完成后，可以评估模型性能：

```bash
python evaluate_ppo.py
```

默认会加载`./models/ppo_final_model.zip`文件和对应的环境标准化参数。

### 直接使用模型

如果你只想在环境中运行训练好的模型：

```bash
python use_trained_model.py
```

### 使用自定义回调

`custom_callbacks.py`文件提供了几种自定义回调函数，可以用于训练过程中的可视化和监控：

```python
from custom_callbacks import get_callbacks

# 在train_ppo.py中使用
callbacks = get_callbacks(save_path="./models", log_path="./logs", save_freq=10000)
model.learn(total_timesteps=100000, callback=callbacks)
```

## 主要参数说明

### PPO参数

- `learning_rate`: 学习率，通常设置为1e-4至5e-4之间
- `n_steps`: 每次策略更新前收集的环境交互步数
- `batch_size`: 每次梯度更新的批次大小
- `n_epochs`: 每批数据用于更新的迭代次数
- `gamma`: 折扣因子
- `gae_lambda`: GAE(广义优势估计)参数
- `clip_range`: PPO裁剪参数

## 自定义环境

可以根据实际需求修改环境ID和奖励函数，适应不同的飞行任务。

## 性能优化提示

1. 增加训练步数通常会提高性能
2. 调整学习率可以显著影响收敛速度和稳定性
3. 使用环境标准化（VecNormalize）可以提高训练效果
4. 对于复杂任务，考虑使用更复杂的网络架构

## 常见问题

- **环境API错误**: 确保使用的是Gymnasium API而不是旧版gym API
- **训练不稳定**: 尝试减小学习率和增加batch_size
- **奖励稀疏**: 考虑设计更密集的奖励函数或使用好奇心驱动的探索
- **环境连接问题**: 确保X-Plane模拟器正在运行，并且XPlaneConnect插件已正确加载 