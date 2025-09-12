"""
XPlaneGym - OpenAI Gym compatible XPlane flight simulator interface

This package provides an XPlane connection environment that conforms to the OpenAI Gym interface specification,
supporting both discrete and continuous action spaces for reinforcement learning algorithm training.
"""

from gymnasium.envs.registration import register

register(
    id="XPlane-v0",
    entry_point="XPlaneGym.envs:XPlaneEnv",
    max_episode_steps=1000,
)

register(
    id="XPlane-custom-v0",
    entry_point="XPlaneGym.envs:XPlaneCustomEnv",
    max_episode_steps=1000,
)
