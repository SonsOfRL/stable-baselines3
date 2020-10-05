from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from gym import spaces
from stable_baselines3.envs.base_env import SC2Env
import logging
import numpy as np

logger = logging.getLogger(__name__)


class FDZEnv(SC2Env):
    metadata = {'render.modes': ['human']}
    default_settings = {
        'map_name': "FindAndDefeatZerglings",
        'players': [sc2_env.Agent(sc2_env.Race.terran),
                    sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.hard)],
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
        self.zerglings = []
        self._num_step = 0
        self._episode_reward = 0
        self._episode = 0

        # 0 no operation
        # 1-4096 attack move selected coordinate
        self.action_space = spaces.Discrete(4097)

        # [0: x, 1: y, 2: hp]
        self.observation_space = spaces.Box(
            low=0,
            high=64,
            shape=(150,),
            dtype=np.uint8
        )

    def reset(self):
        if self.env is None:
            self.init_env()

        self.marines = []
        self.zerglings = []

        self._episode += 1
        self._num_step = 0
        self._episode_reward = 0

        raw_obs = self.env.reset()[0]
        return self.get_derived_obs(raw_obs)

    def init_env(self):
        args = {**self.default_settings, **self.kwargs}
        self.env = sc2_env.SC2Env(**args)

    def get_derived_obs(self, raw_obs):
        obs = np.zeros((50, 3), dtype=np.uint8)
        marines = self.get_units_by_type(raw_obs, units.Terran.Marine, 1)
        zerglings = self.get_units_by_type(raw_obs, units.Zerg.Zergling, 4)
        self.marines = []
        self.zerglings = []

        if len(zerglings) > 0:
            enemy_on_sight = 1
        else:
            enemy_on_sight = 0

        obs[0] = np.array([enemy_on_sight, enemy_on_sight, enemy_on_sight])

        for i, m in enumerate(marines):
            self.marines.append(m)
            obs[i+1] = np.array([m.x, m.y, m[2]])

        for i, z in enumerate(zerglings):
            self.zerglings.append(z)
            obs[i + 4] = np.array([z.x, z.y, z[2]])

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

    def get_enemy_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.ENEMY]

    def take_action(self, action):
        obs = self.env.step([self.do_nothing()])[0]
        x = np.floor((action - 1) / 64)
        y = (action - 1) % 64
        action_mapped = self.attack_move(obs, x, y)


        raw_obs = self.env.step([action_mapped])[0]
        return raw_obs

    def attack_move(self, obs,  x, y):
        try:                         #TODO SELECT ALL ARMY
            marines = self.get_my_units_by_type(obs, units.Terran.Marine)
            target = (x, y)

            distances = self.get_distances(marines, target)
            marine = marines[np.argmax(distances)]

            return actions.RAW_FUNCTIONS.Attack_pt("now", marine.tag, target)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def do_nothing(self):
        return actions.RAW_FUNCTIONS.no_op()

    def get_distances(self, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

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
