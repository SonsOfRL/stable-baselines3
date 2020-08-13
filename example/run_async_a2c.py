from stable_baselines3 import A2C
from stable_baselines3.common.cmd_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env.async_vec_env import AsyncVecEnv
from stable_baselines3.common.vec_env.vec_transpose import DummyTranspose


if __name__ == "__main__":

    BUFFER_SIZE = 5
    BATCH_SIZE = 16
    N_ENV_PER_CORE = 3
    N_ENVS = 16 * N_ENV_PER_CORE
    FRAME_STACK = 4

    # There already exists an environment generator that will make and wrap atari environments correctly.
    env = make_atari_env('PongNoFrameskip-v4', n_envs=N_ENVS, seed=0,
                         vec_env_cls=AsyncVecEnv,
                         wrapper_kwargs={
                             # Stack 4 frames
                             "frame_stack": FRAME_STACK,
                             "transpose": True
                         },
                         vec_env_kwargs={
                             "batchsize": BATCH_SIZE,
                             "buffer_size": BUFFER_SIZE,
                             "n_env_per_core": N_ENV_PER_CORE
                         })

    # Make it compatible with baseline3
    env = DummyTranspose(env)


    model = A2C('CnnPolicy', env, verbose=1)
    model.learn(total_timesteps=100000)

    env.close()