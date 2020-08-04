""" TODO : Write a proper test class """

import numpy as np
import torch
import gym.spaces as spaces
import gym
import time

from stable_baselines3.common.buffers import MultiSharedRolloutBuffer
from stable_baselines3.common.buffers import SharedRolloutStructure
from stable_baselines3.common.vec_env.async_vec_env import AsyncVecEnv
from stable_baselines3.common.vec_env.async_vec_env import JobTuple
from stable_baselines3.common.async_on_policy_algorithm import AsyncOnPolicyAlgorithm


def test_shared_buffer():
    BUFFER_SIZE = 5
    N_ENVS = 12

    env = gym.make("LunarLander-v2")
    obs_space = env.observation_space
    act_space = env.action_space

    mem = MultiSharedRolloutBuffer(BUFFER_SIZE, obs_space, act_space,
                                   n_envs=N_ENVS)
    [
        print(element.shape, element.dtype) for element in mem.buffer
    ]

    state = env.reset()
    mem.add_obs(state, 3, 5)
    other_state = mem.get_obs(3, 5)
    print(np.all(state == other_state))

    mem.add_last_obs(state, 8)
    print(np.all(state == mem.buffer.last_obs[8]))

    mem.add_act(np.ones((4, 1)), torch.ones(4, 1), torch.ones(4, 1),
                np.arange(4), np.arange(4))
    print(
        np.all(mem.buffer.actions[np.arange(4), np.arange(4)] == np.ones((4, 1)))
    )

    state, reward, done, _ = env.step(0)
    reward = np.array(reward, dtype=np.float32).item()
    mem.add_step(state, float(reward), float(done), 4, 4)
    print(np.all(state == mem.buffer.last_obs[4]))
    print(reward == mem.buffer.rewards[4, 4].item())
    print(done == mem.buffer.dones[4, 4].item())

    mem.get_obs([4, 3, 2], [3, 2, 4])
    mem.get_act(3, 4)


def test_async_env():

    N_ENVS = 12
    BUFFER_SIZE = 5
    BATCH_SIZE = 6

    env_fns = [lambda: gym.make("LunarLander-v2") for i in range(N_ENVS)]
    env = AsyncVecEnv(env_fns, n_env_per_core=3,
                      buffer_size=BUFFER_SIZE, batchsize=BATCH_SIZE)

    jobs = env.reset()
    states = env.sharedbuffer.buffer.observations
    actions = env.sharedbuffer.buffer.actions
    rewards = env.sharedbuffer.buffer.rewards
    print(all(np.any(states[pos, ix] != np.zeros_like(states[pos, ix])) for ix, pos, _ in zip(*jobs)))

    last_obs = env.sharedbuffer.get_last_obs(list(jobs.index))
    env.push_jobs(JobTuple(
        jobs.index,
        [4] * len(jobs.index),
        ["step"] * len(jobs.index)
    ))

    time.sleep(1)
    print(not np.all(last_obs == env.sharedbuffer.get_last_obs(list(jobs.index))))

    env.close()


def test_async_on_policy():

    N_ENVS = 12
    BUFFER_SIZE = 5
    BATCH_SIZE = 6

    env_fns = [lambda: gym.make("LunarLander-v2") for i in range(N_ENVS)]
    env = AsyncVecEnv(env_fns, n_env_per_core=3,
                      buffer_size=BUFFER_SIZE, batchsize=BATCH_SIZE)

    model = AsyncOnPolicyAlgorithm(
        "MlpPolicy",
        env,
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        tensorboard_log=None,
        create_eval_env=False,
        policy_kwargs=None,
        verbose=0,
        seed=None,
        device='auto',
        _init_setup_model=True)

    total_timesteps, callback = model._setup_learn(10000, None, None, -1,
                                                   5, None, True,
                                                   "AsyncOnPolicy")

    model.collect_rollouts(model.env, callback,
                           model.rollout_buffer, model.n_steps)

    print(env.sharedbuffer.buffer.values[:, :, 0])
    print(model.rollout_buffer.values)


if __name__ == "__main__":
    test_async_on_policy()
