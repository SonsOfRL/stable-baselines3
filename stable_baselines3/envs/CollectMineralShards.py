from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from gym import spaces
import logging
import numpy as np
from stable_baselines3.envs.base_env import SC2Env

logger = logging.getLogger(__name__)


class CMSEnv(SC2Env):
    metadata = {'render.modes': ['human']}
    default_settings = {
        'map_name': "CollectMineralShards",
        'players': [sc2_env.Agent(sc2_env.Race.terran)],
        'agent_interface_format': features.AgentInterfaceFormat(
            action_space=actions.ActionSpace.RAW,
            use_raw_units=True,
            raw_resolution=64),
        'realtime': False
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.env = None
        self.marines = []
        self.minerals = []
        self.obs = None

        self._num_step = 0
        self._episode_reward = 0
        self._episode = 0

        # 0 no operation
        # 1-4096 attack-move to selected coordinate with marine (64x64 = 4096)
        # 4097-8192 attack-move to selected coordinate by second marine (4096*2 = 8192)
        self.action_space = spaces.Discrete(8193)
        self.observation_space = spaces.Box(
            low=0,
            high=64,
            shape=(30 * 2,),
            dtype=np.uint8
        )

    def reset(self):
        if self.env is None:
            self.init_env()

        self.marines = []
        self.minerals = []
        self._episode += 1
        self._num_step = 0
        self._episode_reward = 0

        raw_obs = self.env.reset()[0]
        return self.get_derived_obs(raw_obs)

    def init_env(self):
        args = {**self.default_settings, **self.kwargs}
        self.env = sc2_env.SC2Env(**args)

    def get_derived_obs(self, raw_obs):
        self.obs = raw_obs
        _PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL

        obs = np.zeros((30, 2), dtype=np.uint8)
        marines = self.get_units_by_type(raw_obs, units.Terran.Marine, 1)
        self.marines = []

        minerals = [[unit.x, unit.y] for unit in raw_obs.observation.raw_units
                    if unit.alliance == _PLAYER_NEUTRAL]

        for i, Marine in enumerate(marines):
            self.marines.append(Marine)
            obs[i] = np.array([Marine.x, Marine.y])

        for i, mineral in enumerate(minerals):
            self.minerals.append(mineral)
            obs[i+2] = minerals[i]
        return obs.reshape(-1)

    def step(self, action):
        raw_obs = self.take_action(action)
        reward = raw_obs.reward
        obs = self.get_derived_obs(raw_obs)
        self._num_step += 1
        self._episode_reward += reward
        self._total_reward += reward
        done = raw_obs.last()
        info = self.get_info() if done else {}
        # each step will set the dictionary to emtpy
        return obs, reward, done, info

    def take_action(self, action):
        if action == 0:
            action_mapped = actions.RAW_FUNCTIONS.no_op()
        elif action <= 4096:
            x = np.floor((action - 1) / 64)
            y = (action - 1) % 64
            action_mapped = self.attack_move(x, y)
        else:
            action = action - 4096
            x = np.floor((action - 1) / 64)
            y = (action - 1) % 64
            action_mapped = self.attack_move(x, y)

        raw_obs = self.env.step([action_mapped])[0]
        return raw_obs

    def attack_move(self, x, y):
        try:
            marines = self.get_my_units_by_type(self.obs, units.Terran.Marine)
            target = (x, y)
            marine = marines[0]

            return actions.RAW_FUNCTIONS.Attack_pt("now", marine.tag, target)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def do_nothing(self):
        return actions.RAW_FUNCTIONS.no_op()

    def get_my_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF]

    def get_units_by_type(self, obs, unit_type, player_relative=0):
        """
        NONE = 0
        SELF = 1
        ALLY = 2
        NEUTRAL = 3
        ENEMY = 4
        """
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == player_relative]

    def close(self):

        if self.env is not None:
            self.env.close()
        super().close()

    def render(self, mode='human', close=False):
        pass
