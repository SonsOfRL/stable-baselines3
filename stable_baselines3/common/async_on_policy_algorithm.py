import time
from typing import Union, Type, Optional, Dict, Any, List, Tuple, Callable

import gym
import torch as th
import numpy as np
from collections import defaultdict

from stable_baselines3.common import logger
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env.async_vec_env import JobTuple


class AsyncOnPolicyAlgorithm(OnPolicyAlgorithm):
    """
    """

    def __init__(self, forwardsize, backwardsize, *args, **kwargs):

        self.backwardsize = backwardsize
        self.forwardsize = forwardsize
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = RolloutBuffer(self.n_steps, self.observation_space,
                                            self.action_space, self.device,
                                            gamma=self.gamma, gae_lambda=self.gae_lambda,
                                            n_envs=self.n_envs)
        self.policy = self.policy_class(self.observation_space, self.action_space,
                                        self.lr_schedule, use_sde=self.use_sde, device=self.device,
                                        **self.policy_kwargs)  # pytype:disable=not-instantiable
        self.policy = self.policy.to(self.device)

    # ---------------------------------- TODO -------------------------------->

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
        self.vec.sharedbuffer.assign_first_obs(jobtuples.index)

        to_train_indxs = []

        callback.on_rollout_start()

        while len(to_train_indxs) < self.backwardsize:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            obs = self.env.sharedbuffer.get_obs(
                jobtuples.poses,
                jobtuples.index,
            )
            forwardsize = len(jobtuples.indexes)
            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(obs).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
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
                                           jobtuples.indexs
                                           )
            # Create new jobs whose actions are ready
            jobtuples = JobTuple(jobtuples.index,
                                 jobtuples.poses,
                                 ["step"] * forwardsize
                                 )
            self.env.push_jobs(jobtuples)

            # Pull new jobs to forward process
            jobs = []
            while len(jobs) < forwardsize:
                job = self.env.pull_ready_jobs()
                if job.poses == (self.env.sharedbuffer.buffer_size):
                    to_train_indxs.append(job.index)
                    # If we are ready to update stop rollout and push pulled
                    # jobs back to the queue
                    if len(to_train_indxs) == self.backwardsize:
                        self.env.push_jobs(JobTuple(*list(zip(jobs))))
                        break
                else:
                    jobs.append(job)
            jobtuples = JobTuple(*list(zip(jobs)))

            if callback.on_step() is False:
                return False
            self.num_timesteps += forwardsize

        # Fill _last_obs with to_train_indxs so that at the next rollout we
        # can start from there
        self._last_obs = JobTuple(
            to_train_indxs,
            [0] * len(to_train_indxs),
            ["act"] * len(to_train_indxs)
        )

        # Calculate next_values
        with th.no_grad():
            obs = self.env.sharedbuffer.get_last_obs(list(to_train_indxs))
            obs_tensor = th.as_tensor(obs).to(self.device)
            actions, values, log_probs = self.policy.forward(obs_tensor)
        # The line below is a consequence of uncorrected bug
        dones = self.env.sharedbuffer.buffer.dones[-1, to_train_indxs]

        # We are ready to fill the rollout_buffer with shared_memory buffer
        rollout_buffer.fill(self.env.sharedbuffer.buffer, to_train_indxs)
        rollout_buffer.compute_returns_and_advantage(values, dones=dones)

        callback.on_rollout_end()
        return True

    # <--------------------------------- TODO ---------------------------------

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(self,
              total_timesteps: int,
              callback: MaybeCallback = None,
              log_interval: int = 1,
              eval_env: Optional[GymEnv] = None,
              eval_freq: int = -1,
              n_eval_episodes: int = 5,
              tb_log_name: str = "OnPolicyAlgorithm",
              eval_log_path: Optional[str] = None,
              reset_num_timesteps: bool = True) -> 'OnPolicyAlgorithm':
        iteration = 0

        total_timesteps, callback = self._setup_learn(total_timesteps, eval_env, callback, eval_freq,
                                                      n_eval_episodes, eval_log_path, reset_num_timesteps,
                                                      tb_log_name)

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback,
                                                      self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.record("rollout/ep_rew_mean",
                                  safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    logger.record("rollout/ep_len_mean",
                                  safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                logger.record("time/fps", fps)
                logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

    def get_torch_variables(self) -> Tuple[List[str], List[str]]:
        """
        cf base class
        """
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
    # ---------------------------------- TODO -------------------------------->
