from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from stable_baselines3.envs.base_env import SC2Env
from gym import spaces
import logging
import numpy as np
import random

logger = logging.getLogger(__name__)


class BMEnv(SC2Env):
    metadata = {'render.modes': ['human']}
    default_settings = {
        'map_name': "CollectMineralAndGas",
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
        self.SCVs = []
        self.depot = []
        self.cc = []
        self.barracks = []
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
            shape=(5 * 1,),
            dtype=np.uint8
        )

    def reset(self):
        if self.env is None:
            self.init_env()

        self.SCVs = []
        self.depot = []
        self.cc = []
        self.barracks = []

        raw_obs = self.env.reset()[0]
        return self.get_derived_obs(raw_obs)

    def init_env(self):
        args = {**self.default_settings, **self.kwargs}
        self.env = sc2_env.SC2Env(**args)

    def get_derived_obs(self, raw_obs):
        obs = np.zeros((6, 1), dtype=np.uint8)
        SCVs = self.get_units_by_type(raw_obs, units.Terran.SCV, 1)
        depots = self.get_units_by_type(raw_obs, units.Terran.SupplyDepot, 1)
        barracks = self.get_units_by_type(raw_obs, units.Terran.Barracks, 1)

        obs[0] = len(SCVs)
        obs[1] = len(depots)
        obs[2] = len(barracks)
        #obs[3] = (is scv produced ?)
        #obs[4] = (how many marine production ?)
        #obs[5] = (current minerals)



        return obs.reshape(-1)

    def step(self, action):
        raw_obs = self.take_action(action)
        reward = raw_obs.reward
        obs = self.get_derived_obs(raw_obs)
        return obs, reward, raw_obs.last(), {}

    def take_action(self, action, obs):
        if action == 0:
            action_mapped = actions.RAW_FUNCTIONS.no_op()
        elif action == 1:
            action_mapped = self.harvest_minerals(obs)
        elif action == 2:
            action_mapped = self.train_scv(obs)
        elif action == 3:
            action_mapped = self.build_supply_depot(obs)
        elif action == 4:
            action_mapped = self.build_barracks(obs)
        else:
            action_mapped = self.train_marine(obs)

        raw_obs = self.env.step([action_mapped])[0]
        return raw_obs

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
        scvs = self.get_units_by_type(obs, units.Terran.SCV)
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
            distances = self.get_distances(obs, mineral_patches, (scv.x, scv.y))
            mineral_patch = mineral_patches[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                "now", scv.tag, mineral_patch.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_supply_depot(self, obs):
        scvs = self.get_units_by_type(obs, units.Terran.SCV)
        if obs.observation.player.minerals >= 100:
            x = random.randint(0, 64)
            y = random.randint(0, 64)
            supply_depot_xy = (x, y)
            distances = self.get_distances(obs, scvs, supply_depot_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
                "now", scv.tag, supply_depot_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_barracks(self, obs):
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(completed_supply_depots) > 0 and
                obs.observation.player.minerals >= 150 and len(scvs) > 0):
            x = random.randint(0, 64)
            y = random.randint(0, 64)
            barracks_xy = (x, y)
            distances = self.get_distances(obs, scvs, barracks_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Barracks_pt(
                "now", scv.tag, barracks_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def train_marine(self, obs):
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(completed_barrackses) > 0 and obs.observation.player.minerals >= 100
                and free_supply > 0):
            for b in range(len(completed_barrackses)):
                barracks = self.get_my_units_by_type(obs, units.Terran.Barracks)[b]
                if barracks.order_length < 1:
                    return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)

            x = random.randint(0, len(completed_barrackses))
            barracks = self.get_my_units_by_type(obs, units.Terran.Barracks)[x]
            if barracks.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def train_scv(self, obs):
        completed_command_center = self.get_my_completed_units_by_type(
            obs, units.Terran.CommandCenter)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(completed_command_center) > 0 and obs.observation.player.minerals >= 50
                and free_supply > 0):
            command_center = self.get_my_units_by_type(obs, units.Terran.CommandCenter)[0]
            if command_center.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_SCV_quick("now", command_center.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def close(self):

        if self.env is not None:
            self.env.close()
        super().close()

    def render(self, mode='human', close=False):
        pass
