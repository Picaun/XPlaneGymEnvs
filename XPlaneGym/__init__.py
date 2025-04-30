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
    id="XPlane-Discrete-v0",
    entry_point="XPlaneGym.envs:XPlaneDiscreteEnv",
    max_episode_steps=1000,
)

register(
    id="XPlane-Continuous-v0",
    entry_point="XPlaneGym.envs:XPlaneContinuousEnv",
    max_episode_steps=1000,
) 