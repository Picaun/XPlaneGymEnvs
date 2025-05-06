# PPO Algorithm Training Example for XPlaneGym

This example demonstrates how to use the PPO (Proximal Policy Optimization) algorithm from Stable-Baselines3 to train a reinforcement learning agent in the XPlaneGym environment.

## Requirements

First install the necessary dependencies:

```bash
pip install -r requirements.txt
```

Make sure you have the XPlaneGym package installed and the X-Plane simulator is correctly configured.

> **Note**: This example uses the Gymnasium API (successor to gym), so ensure you're using Gymnasium >= 0.28.0. The Gymnasium API has some differences from the older gym version, especially in environment reset and step methods.

## File Description

- `train_ppo.py`: Main PPO training script
- `evaluate_ppo.py`: Script for evaluating trained models
- `custom_callbacks.py`: Custom training callbacks
- `use_trained_model.py`: Script to run the environment with a trained model
- `requirements.txt`: Project dependencies

## Usage

### Training the Model

```bash
python train_ppo.py
```

Training parameters can be adjusted in the script, including:
- Environment ID
- Number of training steps
- Learning rate
- Batch size
- Save frequency, etc.

During training, models will be saved periodically in the `./models` directory, and training logs in the `./logs` directory.

### Evaluating the Model

After training, you can evaluate the model's performance:

```bash
python evaluate_ppo.py
```

By default, it will load the `./models/ppo_final_model.zip` file and the corresponding environment normalization parameters.

### Using the Model Directly

If you just want to run a trained model in the environment:

```bash
python use_trained_model.py
```

### Using Custom Callbacks

The `custom_callbacks.py` file provides several custom callback functions that can be used for visualization and monitoring during training:

```python
from custom_callbacks import get_callbacks

# Use in train_ppo.py
callbacks = get_callbacks(save_path="./models", log_path="./logs", save_freq=10000)
model.learn(total_timesteps=100000, callback=callbacks)
```

## Main Parameter Descriptions

### PPO Parameters

- `learning_rate`: Learning rate, typically set between 1e-4 and 5e-4
- `n_steps`: Number of environment interaction steps to collect before each policy update
- `batch_size`: Batch size for each gradient update
- `n_epochs`: Number of iterations for each batch of data used for updates
- `gamma`: Discount factor
- `gae_lambda`: GAE (Generalized Advantage Estimation) parameter
- `clip_range`: PPO clipping parameter

## Custom Environments

You can modify the environment ID and reward function based on actual requirements, adapting to different flight tasks.

## Performance Optimization Tips

1. Increasing training steps usually improves performance
2. Adjusting the learning rate can significantly affect convergence speed and stability
3. Using environment normalization (VecNormalize) can improve training results
4. For complex tasks, consider using more complex network architectures

## Common Issues

- **Environment API errors**: Make sure you're using the Gymnasium API, not the older gym API
- **Unstable training**: Try reducing learning rate and increasing batch size
- **Sparse rewards**: Consider designing denser reward functions or using curiosity-driven exploration
- **Environment connection issues**: Ensure the X-Plane simulator is running and the XPlaneConnect plugin is properly loaded 