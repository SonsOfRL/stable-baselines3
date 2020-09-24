import gym
import numpy as np

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
        _dir = "logs/logs_baseline"
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


def main():
    env_name = "LunarLanderContinuous-v2"

    env = gym.make(env_name)
    vecenv = make_vec_env(env_name, vec_env_cls=SubprocVecEnv, n_envs=32)

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
                train_freq=1,
                gradient_steps=8,
                learning_starts=100,
                n_episodes_rollout=-1,
                verbose=1,
                policy_kwargs=policy_kwargs,
                tensorboard_log="logs_baseline/",
                device="cuda")
    model.learn(total_timesteps=100000,
                log_interval=5,
                callback=loggcallback,
                tb_log_name=env_name,)


if __name__ == "__main__":
    main()
