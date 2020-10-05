import gym
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from stable_baselines3.envs.base_env import SC2Env
from gym import spaces
import logging
import numpy as np


logger = logging.getLogger(__name__)


class DZBEnv(SC2Env):
    metadata = {'render.modes': ['human']}
    default_settings = {
        'map_name': "DefeatZerglingsAndBanelings",
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
        self.banelings = []
        self.zerglings = []
        # 0 no operation
        # 1~32 move
        # 33~122 attack
        self.action_space = spaces.Discrete(123)
        # [0: x, 1: y, 2: hp]
        self.observation_space = spaces.Box(
            low=0,
            high=64,
            shape=(150,),
            dtype=np.uint8
        )
        self._num_step = 0
        self._episode_reward = 0
        self._total_reward = 0

    def reset(self):
        if self.env is None:
            self.init_env()

        self.marines = []
        self.banelings = []
        self.zerglings = []

        raw_obs = self.env.reset()[0]

        self._episode += 1
        self._num_step = 0
        self._episode_reward = 0

        return self.get_derived_obs(raw_obs)

    def init_env(self):
        args = {**self.default_settings, **self.kwargs}
        self.env = sc2_env.SC2Env(**args)

    def get_derived_obs(self, raw_obs):
        obs = np.zeros((50, 3), dtype=np.uint8)
        marines = self.get_units_by_type(raw_obs, units.Terran.Marine, 1)
        zerglings = self.get_units_by_type(raw_obs, units.Zerg.Zergling, 4)
        banelings = self.get_units_by_type(raw_obs, units.Zerg.Baneling, 4)
        self.marines = []
        self.banelings = []
        self.zerglings = []

        for i, m in enumerate(marines):
            self.marines.append(m)
            obs[i] = np.array([m.x, m.y, m[2]])

        for i, b in enumerate(banelings):
            self.banelings.append(b)
            obs[i + len(marines)] = np.array([b.x, b.y, b[2]])

        for i, z in enumerate(zerglings):
            self.zerglings.append(z)
            obs[i + len(marines)+len(banelings)] = np.array([z.x, z.y, z[2]])

        return obs.reshape(-1)

    def step(self, action):
        raw_obs = self.take_action(action)
        reward = raw_obs.reward
        obs = self.get_derived_obs(raw_obs)
        done = raw_obs.last()

        self._num_step += 1
        self._episode_reward += reward
        self._total_reward += reward
        info = self.get_info() if done else {}
        # each step will set the dictionary to emtpy
        return obs, reward, done, info

    def take_action(self, action):
        if action == 0:
            action_mapped = actions.RAW_FUNCTIONS.no_op()
        elif action <= 32:
            derived_action = np.floor((action - 1) / 8)
            idx = (action - 1) % 8
            if derived_action == 0:
                action_mapped = self.move_up(idx)
            elif derived_action == 1:
                action_mapped = self.move_down(idx)
            elif derived_action == 2:
                action_mapped = self.move_left(idx)
            else:
                action_mapped = self.move_right(idx)
        else:
            eidx = np.floor((action - 33) / 9)
            aidx = (action - 33) % 9
            action_mapped = self.attack(aidx, eidx)

        raw_obs = self.env.step([action_mapped])[0]
        return raw_obs

    def move_up(self, idx):
        idx = np.floor(idx)
        try:
            selected = self.marines[idx]
            new_pos = [selected.x, selected.y - 2]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def move_down(self, idx):
        try:
            selected = self.marines[idx]
            new_pos = [selected.x, selected.y + 2]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def move_left(self, idx):
        try:
            selected = self.marines[idx]
            new_pos = [selected.x - 2, selected.y]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def move_right(self, idx):
        try:
            selected = self.marines[idx]
            new_pos = [selected.x + 2, selected.y]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def attack(self, aidx, eidx):
        try:
            selected = self.marines[aidx]
            if eidx > 3:
                # attack zerglings
                target = self.zerglings[eidx - 4]
            else:
                target = self.banelings[eidx]
            return actions.RAW_FUNCTIONS.Attack_unit("now", selected.tag, target.tag)
        except:
            return actions.RAW_FUNCTIONS.no_op()

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
