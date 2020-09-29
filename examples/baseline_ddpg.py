import gym
import numpy as np
import argparse

from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import (NormalActionNoise,
                                            OrnsteinUhlenbeckActionNoise,
                                            VectorizedActionNoise)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import BaseCallback


class LoggerCallback(BaseCallback):

    def __init__(self, _format, log_on_start=None, suffix=""):
        super().__init__()
        self._format = _format
        self.suffix = suffix
        if log_on_start is not None and not isinstance(log_on_start, (list, tuple)):
            log_on_start = tuple(log_on_start)
        self.log_on_start = log_on_start

    def _on_training_start(self) -> None:

        _logger = self.globals["logger"].Logger.CURRENT
        _dir = _logger.dir
        log_format = logger.make_output_format(self._format, _dir, self.suffix)
        _logger.output_formats.append(log_format)
        if self.log_on_start is not None:
            for pair in self.log_on_start:
                _logger.record(*pair, ("tensorboard", "stdout"))

    def _on_step(self) -> bool:
        """
        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True


def main(train_freq, gradient_steps, batch_size, envname, n_envs, log_interval):
    envname = "LunarLanderContinuous-v2"

    env = gym.make(envname)
    vecenv = make_vec_env(envname, vec_env_cls=SubprocVecEnv, n_envs=n_envs)

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    base_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    action_noise = VectorizedActionNoise(base_noise, vecenv.num_envs)

    policy_kwargs = {
        # "actor_arch": [32],
        # "critic_arch": [300, 400],
        "net_arch": [300, 400]
    }

    loggcallback = LoggerCallback("json")

    model = TD3("MlpPolicy",
                vecenv,
                action_noise=action_noise,
                batch_size=batch_size,
                train_freq=train_freq,
                gradient_steps=gradient_steps,
                learning_starts=100,
                n_episodes_rollout=-1,
                verbose=1,
                policy_kwargs=policy_kwargs,
                tensorboard_log="logs_baseline/",
                device="cuda")
    model.learn(total_timesteps=100000,
                log_interval=log_interval,
                callback=loggcallback,
                tb_log_name=envname,)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-freq", help="Number of environment steps between each update loop", default="1", type=int, required=False)
    parser.add_argument("--gradient-steps", help="Update steps per rollout call", default="1", type=int, required=False)
    parser.add_argument("--batch_size", help="Batchsize of each update", default="32", type=int, required=False)
    parser.add_argument("--envname", help="Gym environment", default="LunarLanderContinuous-v2", type=str, required=False)
    parser.add_argument("--n_envs", help="Parallel environments in synch rollout gathering", default="1", type=int, required=False)
    parser.add_argument("--log_interval", help="Logging interval between update calls", default=5, type=int)
    kwargs = vars(parser.parse_args())
    main(**kwargs)
