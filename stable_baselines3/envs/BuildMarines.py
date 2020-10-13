from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from stable_baselines3.envs.base_env import SC2Env
from gym import spaces
import logging
import numpy as np
import random
from pysc2.agents import base_agent


logger = logging.getLogger(__name__)


class BMEnv(SC2Env):
    metadata = {'render.modes': ['human']}
    default_settings = {
        'map_name': "BuildMarines",
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
        self.obs = None

        self._num_step = 0
        self._episode_reward = 0
        self._total_reward = 0
    # 0 no operation
        # 1 harvest minerals
        # 2 train SCV's
        # 3 build supply depot
        # 4 build barracks
        # 5 train marines

        self.action_space = spaces.Discrete(6)
        # [0: x, 1: y, 2: hp]
        self.observation_space = spaces.Box(
            low=0,
            high=64,
            shape=(12 * 1,),
            dtype=np.uint8
        )

    def reset(self):
        if self.env is None:
            self.init_env()

        raw_obs = self.env.reset()[0]

        self._episode += 1
        self._num_step = 0
        self._episode_reward = 0
        return self.get_derived_obs(raw_obs)

    def init_env(self):
        args = {**self.default_settings, **self.kwargs}
        self.env = sc2_env.SC2Env(**args)

    def get_derived_obs(self, raw_obs):
        self.obs = raw_obs
        new_obs = np.zeros((12, 1), dtype=np.uint8)
        SCVs = self.get_units_by_type(raw_obs, units.Terran.SCV, 1)
        depots = self.get_units_by_type(raw_obs, units.Terran.SupplyDepot, 1)
        completed_depots = self.get_my_completed_units_by_type(raw_obs, units.Terran.SupplyDepot)
        barracks = self.get_units_by_type(raw_obs, units.Terran.Barracks, 1)
        completed_barracks = self.get_my_completed_units_by_type(raw_obs, units.Terran.Barracks)

        idle_scvs = [SCV for SCV in SCVs if SCV.order_length == 0]
        queued_marines = (completed_barracks[0].order_length
        if len(completed_barracks) > 0 else 0)
        free_supply = (raw_obs.observation.player.food_cap -
                                 raw_obs.observation.player.food_used)
        can_afford_supply_depot = raw_obs.observation.player.minerals >= 100
        can_afford_barracks = raw_obs.observation.player.minerals >= 150
        can_afford_marine = raw_obs.observation.player.minerals >= 100

        new_obs[0] = raw_obs.observation.player.minerals
        new_obs[1] = len(SCVs)
        new_obs[2] = len(depots)
        new_obs[3] = len(completed_depots)
        new_obs[4] = len(barracks)
        new_obs[5] = len(completed_barracks)
        new_obs[6] = len(idle_scvs)
        new_obs[7] = free_supply
        new_obs[8] = can_afford_supply_depot
        new_obs[9] = can_afford_barracks
        new_obs[10] = can_afford_marine
        new_obs[11] = queued_marines

        return new_obs.reshape(-1)

    def step(self, action):
        raw_obs = self.take_action(action)
        reward = raw_obs.reward
        obs = self.get_derived_obs(raw_obs)
        done = raw_obs.last()

        self._num_step += 1
        self._episode_reward += reward
        self._total_reward += reward

        info = self.get_info() if done else {}
        return obs, reward, done, info

    def take_action(self, action):
        if action == 0:
            action_mapped = actions.RAW_FUNCTIONS.no_op()
        elif action == 1:
            action_mapped = self.harvest_minerals(self.obs)
        elif action == 2:
            action_mapped = self.train_scv(self.obs)
        elif action == 3:
            action_mapped = self.build_supply_depot(self.obs)
        elif action == 4:
            action_mapped = self.build_barracks(self.obs)
        else:
            action_mapped = self.train_marine(self.obs)

        raw_obs = self.env.step([action_mapped])[0]
        return raw_obs

    def do_nothing(self):
        return actions.RAW_FUNCTIONS.no_op()

    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

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

    def get_my_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF]

    def get_my_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.SELF]

    def harvest_minerals(self, obs):
        scvs = self.get_my_completed_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        if len(idle_scvs) > 0:
            mineral_patches = [unit for unit in obs.observation.raw_units
                               if unit.unit_type in [
                                   units.Neutral.BattleStationMineralField,
                                   units.Neutral.BattleStationMineralField750,
                                   units.Neutral.LabMineralField,
                                   units.Neutral.LabMineralField750,
                                   units.Neutral.MineralField,
                                   units.Neutral.MineralField750,
                                   units.Neutral.PurifierMineralField,
                                   units.Neutral.PurifierMineralField750,
                                   units.Neutral.PurifierRichMineralField,
                                   units.Neutral.PurifierRichMineralField750,
                                   units.Neutral.RichMineralField,
                                   units.Neutral.RichMineralField750
                               ]]
            scv = random.choice(idle_scvs)
            if len(mineral_patches) > 0:
                mineral_patch = random.choice(mineral_patches)
                return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                    "now", scv.tag, mineral_patch.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_supply_depot(self, obs):
        scvs = self.get_my_completed_units_by_type(obs, units.Terran.SCV)
        if obs.observation.player.minerals >= 100:
            x = random.randint(0, 64)
            y = random.randint(0, 64)
            supply_depot_xy = (x, y)

            scv = random.choice(scvs)
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
                "now", scv.tag, supply_depot_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_barracks(self, obs):
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        scvs = self.get_my_completed_units_by_type(obs, units.Terran.SCV)
        if (len(completed_supply_depots) > 0 and
                obs.observation.player.minerals >= 150 and len(scvs) > 0):
            x = random.randint(0, 64)
            y = random.randint(0, 64)
            barracks_xy = (x, y)
            scv = random.choice(scvs)
            return actions.RAW_FUNCTIONS.Build_Barracks_pt(
                "now", scv.tag, barracks_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def train_marine(self, obs):
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(completed_barrackses) > 0 and obs.observation.player.minerals >= 50
                and free_supply > 0):
            for b in range(len(completed_barrackses)-1):
                barracks = self.get_my_completed_units_by_type(obs, units.Terran.Barracks)[b]
                if barracks.order_length < 1:
                    return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)

            x = random.randint(0, len(completed_barrackses)-1)
            barracks = self.get_my_completed_units_by_type(obs, units.Terran.Barracks)[x]
            if barracks.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def train_scv(self, obs):
        scvs = self.get_my_completed_units_by_type(obs, units.Terran.SCV)
        completed_command_center = self.get_my_completed_units_by_type(
            obs, units.Terran.CommandCenter)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(completed_command_center) > 0 and obs.observation.player.minerals >= 50
                and free_supply > 0):
            command_center = self.get_my_completed_units_by_type(obs, units.Terran.CommandCenter)[0]
            if command_center.order_length < 2:
                return actions.RAW_FUNCTIONS.Train_SCV_quick("now", command_center.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def close(self):

        if self.env is not None:
            self.env.close()
        super().close()

    def render(self, mode='human', close=False):
        pass
