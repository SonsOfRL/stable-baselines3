import os
import yaml
import subprocess
import torch as th

import stable_baselines3
from stable_baselines3 import A2C
from stable_baselines3.common.cmd_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy


def read_hypers():
    with open(f"hyperparams.yaml", "r") as f:
        hyperparams_dict = yaml.safe_load(f)
        return hyperparams_dict["atari"]

class PolicyCallback(BasePolicy):
    def __init__(self):
        super().__init__()


    def init_weights(self, module: th.nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        hyperparams = read_hypers()

        for _ in hyperparams:

            __, hyperparam = list(hyperparams.items())[0]

        if hyperparam['inits'] == 'orthogonal':
            if isinstance(module, (th.nn.Linear, th.nn.Conv2d)):
                th.nn.init.orthogonal_(module.weight, gain=gain)
                if module.bias is not None:
                    module.bias.data.fill_(0.0)

        elif hyperparam['inits'] == 'xavier_normal':
            if isinstance(module, th.nn.Linear):
                th.nn.init.xavier_normal_(module.weight, gain)
                th.nn.init.zeros_(module.bias)

        elif hyperparam['inits'] == 'xavier_uniform':
            if isinstance(module, th.nn.Linear):
                th.nn.init.xavier_uniform_(module.weight, gain)
                th.nn.init.zeros_(module.bias)

        elif hyperparam['inits'] == 'kaiming_uniform':
            if isinstance(module, th.nn.Linear):
                th.nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                th.nn.init.zeros_(module.bias)

        elif hyperparam['inits'] == 'kaiming_normal':
            if isinstance(module, th.nn.Linear):
                th.nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                th.nn.init.zeros_(module.bias)

        elif hyperparam['inits'] == 'dirac':
            if isinstance(module, th.nn.Conv2d):
                th.nn.init.dirac_(module.weight)
                th.nn.init.zeros_(module.bias)

        elif hyperparam['inits'] == 'sparse':
            if isinstance(module, th.nn.Conv2d):
                th.nn.init.sparse_(module.weight, sparsity=0.1)
                th.nn.init.zeros_(module.bias)

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


if __name__ == "__main__":

    hyperparams = read_hypers()

    path = "/" + os.path.join(*stable_baselines3.__file__.split("/")[:-2])
    #commit_num = subprocess.check_output(["git", "describe", "--always"], cwd=path).strip().decode()

    for atarigame in hyperparams:

        atariname, hyperparam = list(hyperparams.items())[0]

        loggcallback = LoggerCallback(
            "json",
            [("hypers", hyperparam)
            ]
        )
        ##("commit", commit_num)
        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                            net_arch=hyperparam['net_arch'])

        env = make_atari_env(hyperparam["envname"],
                             vec_env_cls=SubprocVecEnv,
                             **hyperparam["env"])
        env = VecFrameStack(env, **hyperparam["framestack"])

        model = A2C(env=env,
                    verbose=1,
                    tensorboard_log="logs",
                    policy_kwargs=policy_kwargs,
                    **hyperparam["agent"])

        model.learn(callback=loggcallback,
                    tb_log_name=atariname,
                    **hyperparam["learn"])
