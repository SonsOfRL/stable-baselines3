from stable_baselines3 import A2C
from stable_baselines3.common.cmd_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


if __name__ == "__main__":

    # There already exists an environment generator that will make and wrap atari environments correctly.
    env = make_atari_env('PongNoFrameskip-v4', n_envs=4, seed=0, vec_env_cls=SubprocVecEnv)
    # Stack 4 frames
    env = VecFrameStack(env, n_stack=4)

    model = A2C('CnnPolicy', env, verbose=0)
    model.learn(total_timesteps=10000)
