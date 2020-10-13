from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from gym import spaces
import logging
import numpy as np
from stable_baselines3.envs.base_env import SC2Env

logger = logging.getLogger(__name__)


class MTBEnv(SC2Env):
    metadata = {'render.modes': ['human']}
    default_settings = {
        'map_name': "MoveToBeacon",
        'players': [sc2_env.Agent(sc2_env.Race.terran)],
        'agent_interface_format': features.AgentInterfaceFormat(
            action_space=actions.ActionSpace.RAW,
            use_raw_units=True,
            raw_resolution=64,
            use_feature_units=True),
        'realtime': False
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.env = None
        self.marines = []
        self.beacon = []
        self._num_step = 0
        self._episode_reward = 0
        self._episode = 0
        self.obs = None

        # 0 no operation
        # 1-4096 attack-move to selected coordinate by first marine (64x64 = 4096)
        self.action_space = spaces.Discrete(4097)
        self.observation_space = spaces.Box(
            low=0,
            high=64,
            shape=(2 * 2,),
            dtype=np.uint8
        )

    def reset(self):
        if self.env is None:
            self.init_env()

        self.marines = []
        self.beacon = []
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
        beacon = self.get_beacon(raw_obs)
        marine = self.get_units_by_type(raw_obs, units.Terran.Marine, 1)
        obs = np.zeros((2, 2), dtype=np.uint8)

        obs[0] = [marine[0].x, marine[0].y]
        obs[1] = [beacon[0].x, beacon[0].y]
        return obs.reshape(-1)

    def get_neutral_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.NEUTRAL]

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
        else:
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

    def get_beacon(self, obs):
        return [unit for unit in obs.observation.raw_units
               if unit.alliance == features.PlayerRelative.NEUTRAL]

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
