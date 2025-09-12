# XPlane Gym：兼容 X-Plane 的强化学习环境

XPlaneGymEnvs 是一个符合 OpenAI Gym 接口的 X-Plane 飞行模拟器环境，专门为强化学习研究设计。它可以与 X-Plane 模拟器无缝集成，支持离散和连续动作空间，可用于训练智能体执行飞行控制任务。

## 安装要求

* X-Plane 12 飞行模拟器（目前只在此版本测试过）
* Python 3.8+
* gymnasium
* numpy

## 安装步骤

```bash
# 克隆仓库
git clone https://github.com/Picaun/XPlaneGymEnvs.git
cd XPlaneGymEnvs

# 安装项目及依赖
pip install -e .
```

## 可用环境

* `XPlane-v0`：基础环境，可配置为离散或连续动作空间
* `XPlane-custom-v0`：自定义环境，通过``` import XPlaneEnv ```类来自定义环境

## 快速开始

### 1. 安装

```bash
pip install -e .
```

### 2. 启动 X-Plane

1. 打开 X-Plane 飞行模拟器
2. 确认在 **设置 > 网络** 中，UDP 通信端口设为 `49000`（默认值）
3. 开始新飞行，可自行设置天气和飞行时间
   （注意：目前 XPlaneGymEnvs 接口还不能直接控制 X-Plane 12 的这些功能）

### 3. 使用示例智能体

```bash
cd agent_examples/dqn_example
python train_dqn.py
```

## 环境参数配置

创建环境时可以配置多种参数：

```python
env = gym.make(
    "XPlane-Continuous-v0",
    ip='127.0.0.1',                # X-Plane IP 地址
    port=49000,                    # X-Plane UDP 端口
    timeout=1.0,                   # 通信超时时间
    pause_delay=0.05,              # 动作执行延迟
    starting_latitude=37.558,      # 初始纬度（默认首尔金浦国际机场附近）
    starting_longitude=126.790,    # 初始经度（默认首尔金浦国际机场附近）
    starting_altitude=3000.0,      # 初始高度
    starting_velocity=100.0,       # 初始速度
    starting_pitch_range=10.0,     # 初始俯仰角随机范围
    starting_roll_range=20.0,      # 初始横滚角随机范围
    random_desired_state=True,     # 是否随机目标姿态
    desired_pitch_range=5.0,       # 目标俯仰角随机范围
    desired_roll_range=10.0,       # 目标横滚角随机范围
    action_dim=2                   # 连续动作空间维度
)
```

## 环境接口说明

### 观测空间（Observation Space）

默认观测空间包含 12 个连续值：

1. 横滚角偏差（度）
2. 俯仰角偏差（度）
3. 航向角（度）
4. 横滚角速度（弧度/秒）
5. 俯仰角速度（弧度/秒）
6. 偏航角速度（弧度/秒）
7. 攻角（度）
8. 侧滑角（度）
9. 离地高度（米）
10. 纬度（度）
11. 经度（度）
12. 海拔高度（米）

### 离散动作空间

默认提供 9 种离散动作：
0\. 无动作

1. 俯仰上升
2. 俯仰下降
3. 横滚左转
4. 横滚右转
5. 俯仰上升 + 横滚左转
6. 俯仰上升 + 横滚右转
7. 俯仰下降 + 横滚左转
8. 俯仰下降 + 横滚右转

### 连续动作空间

根据配置可提供 2-4 维连续动作：

* 2D: \[俯仰控制, 横滚控制]
* 3D: \[俯仰控制, 横滚控制, 方向舵控制]
* 4D: \[俯仰控制, 横滚控制, 方向舵控制, 油门]

各维度范围为 \[-1.0, 1.0]，油门范围为 \[0.0, 1.0]。

### 奖励函数（Reward Function）

奖励函数主要基于当前姿态与目标姿态的偏差计算，包括：

* 姿态接近目标状态的奖励
* 姿态偏差过大的惩罚
* 低高度惩罚
* 攻角过大惩罚
* 飞机坠毁的负奖励
* 连续环境中的平滑控制奖励

## 自定义扩展

你可以通过继承基础类来扩展环境功能：

* 继承 `XPlaneEnv` 创建新的环境类
* 重写 `_compute_reward` 方法自定义奖励函数
* 重写 `_get_observation` 方法自定义观测空间

## 使用示例

```python
import gymnasium as gym
import XPlaneGym
import numpy as np

# 创建环境
env = gym.make("XPlane-v0")

# 重置环境，获取初始观测值
observation, info = env.reset()

# 运行 10 个回合
for episode in range(10):
    observation, info = env.reset()
    episode_reward = 0
    step_count = 0
    done = False
    
    print(f"开始第 {episode+1} 回合")
    
    # 单回合循环
    while not done and step_count < 100:
        # 随机采样动作（实际应用应使用策略网络）
        action = env.action_space.sample()
        
        # 执行动作
        observation, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        step_count += 1
        done = terminated or truncated
        
        # 打印当前状态
        print(f"步骤 {step_count}: 即时奖励 = {reward:.2f}, 累积奖励 = {episode_reward:.2f}")
        print(f"俯仰角偏差: {observation[1]:.2f}°, 横滚角偏差: {observation[0]:.2f}°")
        print(f"位置: 纬度 {observation[9]:.6f}°, 经度 {observation[10]:.6f}°, 高度 {observation[11]:.2f}m")
        
    print(f"回合 {episode+1} 结束: 总步数 = {step_count}, 总奖励 = {episode_reward:.2f}")
    print("-" * 50)

# 关闭环境
env.close()
```

## Issue

由于个人能力有限，在运行过程中难免会出现各种问题。幸运的是，我们可以通过 issue 进行交流和解决。

## 许可证

MIT

## 致谢

本项目基于以下开源项目：

* [XPlaneConnectX](https://github.com/sisl/XPlaneConnectX)
* [GYM\_XPLANE\_ML](https://github.com/adderbyte/GYM_XPLANE_ML)
* [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)

---

要不要我帮你直接生成一个 `README_zh.md` 文件，和原来的英文 `README.md` 并列放在项目里？
