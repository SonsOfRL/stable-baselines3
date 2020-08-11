from stable_baselines3 import A2C
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

from examples.StarCraft2.envs.MoveToBeacon import MoveToBeaconEnv


if __name__ == "__main__":
    env = MoveToBeaconEnv()
    check_env(env)
    env = make_vec_env(lambda: env, n_envs=1)

    model = A2C('CnnPolicy', env, verbose=1)
    model.learn(total_timesteps=1e5)
