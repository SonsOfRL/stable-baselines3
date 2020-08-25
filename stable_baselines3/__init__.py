import os

from stable_baselines3.a2c import A2C
from stable_baselines3.ddpg import DDPG
from stable_baselines3.dqn import DQN
from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC
from stable_baselines3.td3 import TD3
from gym.envs.registration import register

register(
    id='defeat-roaches-v0',
    entry_point='stable-baselines3.envs:DREnv',
)
register(
    id='defeat-zerglings-banelings-v0',
    entry_point='stable-baselines3.envs:DZBEnv',
)
register(
    id='move-to-beacon-v0',
    entry_point='stable-baselines3.envs:MTBEnv',
)
register(
    id='collect-mineral-shards-v0',
    entry_point='stable-baselines3.envs:CMSEnv',
)
register(
    id='find-and-defeat-zerglings-v0',
    entry_point='stable-baselines3.envs:FDZEnv',
)
register(
    id='collect-mineral-and-gas-v0',
    entry_point='stable-baselines3.envs:CMGEnv',
)
register(
    id='build-marines-v0',
    entry_point='stable-baselines3.envs:BMEnv',
)
# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file, "r") as file_handler:
    __version__ = file_handler.read().strip()
