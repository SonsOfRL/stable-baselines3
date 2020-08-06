import time
from typing import Union, Type, Optional, Dict, Any, List, Tuple, Callable

import gym
import torch as th
import numpy as np
from collections import defaultdict

from stable_baselines3.common import logger
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env.async_vec_env import JobTuple
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm


class AsyncOnPolicyAlgorithm(OnPolicyAlgorithm):
    """
    """

    def collect_rollouts(self,
                         env: VecEnv,
                         callback: BaseCallback,
                         rollout_buffer: RolloutBuffer,
                         n_rollout_steps: int) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        # n_steps = 0
        rollout_buffer.reset()
        # Rollouts in the Rollot buffer
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        # Instead of states, _last_obs contain job information about the
        # lastly updated environments
        jobtuples = self._last_obs
        batchsize = len(jobtuples.index)
        self.env.sharedbuffer.assign_first_obs(list(jobtuples.index))

        to_train_indxs = []

        callback.on_rollout_start()

        while len(to_train_indxs) < batchsize:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            obs = self.env.sharedbuffer.get_obs(
                jobtuples.poses,
                jobtuples.index,
            )
            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(obs).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)

            if len(log_probs.shape) <= 1:
                # Reshape 0-d and 1-d tensors to avoid error
                log_probs = log_probs.reshape(-1, 1)

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions,
                                          self.action_space.low,
                                          self.action_space.high)
            self.env.sharedbuffer.add_act(actions,
                                          values,
                                          log_probs,
                                          jobtuples.poses,
                                          jobtuples.index
                                          )
            # Create new jobs whose actions are ready
            jobtuples = JobTuple(jobtuples.index,
                                 jobtuples.poses,
                                 ["step"] * batchsize
                                 )
            self.env.push_jobs(jobtuples)

            # Pull new jobs to forward process
            jobs = []
            while len(jobs) < batchsize:
                job = self.env.pull_ready_jobs()
                if job.poses == (self.env.sharedbuffer.buffer_size):
                    to_train_indxs.append(job.index)
                    # If we are ready to update stop rollout and push pulled
                    # jobs back to the queue
                    if len(to_train_indxs) == batchsize:
                        if len(jobs) > 0:
                            self.env.re_push_ready_jobs(jobs)
                        break
                else:
                    jobs.append(job)
            else:
                jobtuples = JobTuple(*zip(*jobs))
                self.num_timesteps += batchsize

            if callback.on_step() is False:
                return False

        # Fill _last_obs with to_train_indxs so that at the next rollout we
        # can start from there
        self._last_obs = JobTuple(
            to_train_indxs,
            [0] * len(to_train_indxs),
            ["act"] * len(to_train_indxs)
        )

        # We are ready to fill the rollout_buffer with shared_memory buffer
        last_obs = self.env.fill(rollout_buffer, to_train_indxs)

        # Calculate next_values
        with th.no_grad():
            obs_tensor = th.as_tensor(last_obs).to(self.device)
            _, last_val, _ = self.policy.forward(obs_tensor)
        # The line below is a consequence of uncorrected bug
        dones = self.env.sharedbuffer.buffer.dones[-1, to_train_indxs].flatten()

        rollout_buffer.compute_returns_and_advantage(last_val, dones=dones)

        callback.on_rollout_end()
        return True
