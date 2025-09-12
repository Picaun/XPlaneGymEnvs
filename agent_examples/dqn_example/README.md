# DQN Algorithm Training Example for XPlaneGym

This example demonstrates how to use the DQN (Deep Q-Network) algorithm from Stable-Baselines3 to train a reinforcement learning agent in the XPlaneGym environment.

## Environment Requirements

First, install the necessary dependencies:

```bash
pip install -r requirements.txt
```

Make sure XPlaneGym package is installed and the X-Plane simulator is correctly configured.

> **Note**: This example uses the Gymnasium (successor to gym) API. Please ensure you're using Gymnasium >= 0.28.0. The Gymnasium API has some differences from the old gym, especially in environment reset and step methods.

## File Description

- `train_dqn.py`: Main DQN training script
- `evaluate_dqn.py`: Script to evaluate trained models
- `use_trained_model.py`: Run environment with a trained model
- `custom_callbacks.py`: Custom training callback functions
- `requirements.txt`: Project dependencies

## DQN Algorithm Introduction

DQN (Deep Q-Network) is a reinforcement learning algorithm that combines Q-learning with deep neural networks. It has the following features:

1. **Experience Replay**: Stores and reuses past experiences, reducing correlation between samples
2. **Target Network**: Uses a separate network to calculate target Q-values, improving training stability
3. **Deep Neural Network**: Uses neural networks to approximate the Q-function, handling high-dimensional state spaces

## Usage

### Training the Model

```bash
python train_dqn.py
```

Training parameters can be adjusted in the script, including:
- Environment ID
- Learning rate
- Exploration rate
- Experience replay buffer size
- Discount factor
- Target network update frequency, etc.

During training, the model is periodically saved in the `./models` directory, and training logs are saved in the `./logs` directory.

### Resuming Training (Checkpoint Resume)

Support resuming training from the last checkpoint and restoring VecNormalize statistics.

Basic resume (auto-load the latest `dqn_model*_steps.zip` and `vec_normalize.pkl` from `--save_path`):

```bash
python train_dqn.py --resume true
```

Specify a particular checkpoint file:

```bash
python train_dqn.py --resume true --checkpoint_path ./models/dqn_model_100000_steps.zip
```

Notes:
- `vec_normalize.pkl` will be auto-loaded/saved in `--save_path` for normalization continuity.
- Remaining timesteps are computed as `total_timesteps - already_trained`, with `reset_num_timesteps` handled appropriately.

### Evaluating the Model

After training, you can evaluate model performance:

```bash
python evaluate_dqn.py
```

By default, it loads the `./models/dqn_final_model.zip` file.

### Directly Using the Model

If you just want to run a trained model in the environment:

```bash
python use_trained_model.py
```

### Using Custom Callbacks

The `custom_callbacks.py` file provides several custom callback functions that can be used for visualization and monitoring during training:

```python
from custom_callbacks import get_callbacks

# Use in train_dqn.py
callbacks = get_callbacks(save_path="./models", log_path="./logs", save_freq=10000)
model.learn(total_timesteps=100000, callback=callbacks)
```

## Key Parameter Descriptions

### DQN Key Parameters

- `learning_rate`: Learning rate, typically from 1e-4 to 1e-3
- `buffer_size`: Experience replay buffer size, larger stores more samples
- `learning_starts`: Number of samples to collect before learning begins
- `batch_size`: Batch size for each gradient update
- `gamma`: Discount factor, controls importance of future rewards
- `target_update_interval`: Target network update frequency
- `train_freq`: Training frequency
- `exploration_fraction`: Exploration ratio, controls exploration rate decay speed
- `exploration_initial_eps`: Initial exploration rate
- `exploration_final_eps`: Final exploration rate

## Custom Environment

You can modify the environment ID and reward function according to actual needs, adapting to different flight tasks.

## Performance Optimization Tips

1. **Increase Buffer Size**: A larger buffer can store more experiences but increases memory usage
2. **Adjust Target Network Update Frequency**: Lower update frequency increases stability but may slow convergence
3. **Exploration Rate Settings**: Initial high exploration rate helps discover new strategies, then gradually decreases
4. **Reward Scaling**: Appropriate scaling of reward values can improve training stability

## Common Issues

- **Memory Overflow**: Try reducing experience replay buffer size
- **Unstable Training**: Adjust target network update frequency or reduce learning rate
- **Overfitting**: Increase training data or add regularization
- **Action Jittering**: May be due to exploration rate set too high or unstable Q-value estimates 
