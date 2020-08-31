import os
import yaml
import subprocess
import stable_baselines3
from stable_baselines3.envs.DefeatZerglingsAndBanelings import DZBEnv
from stable_baselines3.envs.DefeatRoaches import DREnv
from stable_baselines3.envs.CollectMineralAndGas import CMGEnv
from stable_baselines3.envs.CollectMineralShards import CMSEnv
from stable_baselines3.envs.FindAndDefeatZerglings import FDZEnv
from stable_baselines3.envs.MoveToBeacon import MTBEnv
from stable_baselines3.envs.BuildMarines import BMEnv
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.cmd_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack




def read_hypers():
    with open(f"Starcraft_hyper.yaml", "r") as f:
        hyperparams_dict = yaml.safe_load(f)
        return hyperparams_dict["starcraft"]


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

    #path = "/" + os.path.join(*stable_baselines3.__file__.split("/")[:-2])
    #commit_num = subprocess.check_output(["git", "describe", "--always"], cwd=path).strip().decode()

    for starcraftgame in hyperparams:

        gamename, hyperparam = list(starcraftgame.items())[0]

        loggcallback = LoggerCallback(
            "json",
            [("hypers", hyperparam)]
        )


        env = DummyVecEnv([lambda: DZBEnv()])

        #env = make_atari_env(hyperparam["envname"],
                           #  vec_env_cls=dummyvecenv,
                           #  **hyperparam["env"])

        #env = VecFrameStack(env, **hyperparam["framestack"])

        model = A2C(env=env,
                    verbose=1,
                    tensorboard_log="logs",
                    **hyperparam["agent"])

        model.learn(callback=loggcallback,
                    tb_log_name=gamename,
                    **hyperparam["learn"])