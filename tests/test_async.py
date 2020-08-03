""" TODO : Write a proper test class """

import numpy as np
import torch
import gym.spaces as spaces
import gym

from stable_baselines3.common.buffers import SharedRolloutBuffer
from stable_baselines3.common.buffers import SharedRolloutStructure
from stable_baselines3.common.vec_env.async_vec_env import AsyncVecEnv
from stable_baselines3.common.vec_env.async_vec_env import JobTuple


def test_shared_buffer():
    env = gym.make("LunarLander-v2")
    obs_space = env.observation_space
    act_space = env.action_space

    mem = SharedRolloutBuffer(obs_space, act_space, n_envs=10)
    print(
        mem.buffer.actions.shape,
        mem.buffer.observations.shape,
        mem.buffer.rewards.shape,
    )


def test_async_env():

    N_ENVS = 12
    
    env_fns = [lambda: gym.make("LunarLander-v2") for i in range(N_ENVS)]
    env = AsyncVecEnv(env_fns, n_env_per_core=3)


if __name__ == "__main__":
    test_async_env()
