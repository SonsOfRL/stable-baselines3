import gym
from stable_baselines3.envs.DefeatZerglingsAndBanelings import DZBEnv
from stable_baselines3.envs.DefeatRoaches import DREnv
from stable_baselines3.envs.CollectMineralAndGas import CMGEnv
from stable_baselines3.envs.CollectMineralShards import CMSEnv
from stable_baselines3.envs.FindAndDefeatZerglings import FDZEnv
from stable_baselines3.envs.MoveToBeacon import MTBEnv
from stable_baselines3.envs.BuildMarines import BMEnv
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
from absl import flags

FLAGS = flags.FLAGS
FLAGS([''])

# create vectorized environment
env = gym.make('defeat-zerglings-banelings-v0')
eng = DZBEnv()
env = DummyVecEnv([lambda: DZBEnv()])

# use A2C to learn and save the model when finished
model = A2C(ActorCriticCnnPolicy, env, verbose=1, tensorboard_log="log/")
model.learn(total_timesteps=int(1e5), tb_log_name="first_rum", reset_num_timesteps=False)
model.save("model/dzb_A2C")