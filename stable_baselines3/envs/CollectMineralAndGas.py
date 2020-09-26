from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from stable_baselines3.envs.base_env import SC2Env
from gym import spaces
import logging
import numpy as np
import random

logger = logging.getLogger(__name__)


class CMGEnv(SC2Env):
    metadata = {'render.modes': ['human']}
    default_settings = {
        'map_name': "CollectMineralsAndGas",
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
        self.supply_depot = []
        self.command_center = []
        self.refinery = []

        self._num_step = 0
        self._episode_reward = 0
        self._episode = 0

        # 0 no operation
        # 1~32 move
        # 33~122 attack
        self.action_space = spaces.Discrete(7)
        # [0: x, 1: y, 2: hp]
        self.observation_space = spaces.Box(
            low=0,
            high=64,
            shape=(11 * 1),
            dtype=np.uint8
        )

    def reset(self):
        if self.env is None:
            self.init_env()

        self.SCVs = []
        self.supply_depot = []
        self.command_center = []
        self.refinery = []

        self._episode += 1
        self._num_step = 0
        self._episode_reward = 0

        raw_obs = self.env.reset()[0]
        return self.get_derived_obs(raw_obs)

    def init_env(self):
        args = {**self.default_settings, **self.kwargs}
        self.env = sc2_env.SC2Env(**args)

    def get_derived_obs(self, raw_obs):

        SCVs = self.get_units_by_type(raw_obs, units.Terran.SCV, 1)
        idle_scvs = [scv for scv in SCVs if scv.order_length == 0]
        supply_depot = self.get_units_by_type(raw_obs, units.Terran.SupplyDepot, 1)
        command_center = self.get_units_by_type(raw_obs, units.Terran.CommandCenter, 1)
        refinery = self.get_units_by_type(raw_obs, units.Terran.Refinery, 1)
        minerals = raw_obs.observation.player.minerals
        free_supply =(raw_obs.observation.player.food_cap -
                       raw_obs.observation.player.food_used)
        can_afford_supply_depot = raw_obs.observation.player.minerals >= 100
        can_afford_barracks = raw_obs.observation.player.minerals >= 150
        can_afford_marine = raw_obs.observation.player.minerals >= 100
        can_afford_refinery = raw_obs.observation.player.minerals >= 75

        obs = np.zeros((11, 1), dtype=np.uint8)
        obs[0] = len(SCVs)
        obs[1] = len(supply_depot)
        obs[2] = len(command_center)
        obs[3] = len(refinery)
        obs[4] = can_afford_barracks
        obs[5] = can_afford_supply_depot
        obs[6] = can_afford_marine
        obs[7] = can_afford_refinery
        obs[8] = minerals
        obs[9] = free_supply
        obs[10] = len(idle_scvs)
        return obs.reshape(-1)

    def step(self, action):
        obs = self.env.step([self.do_nothing()])[0]
        raw_obs = self.take_action(obs, action)
        reward = raw_obs.reward
        obs = self.get_derived_obs(raw_obs)
        self._num_step += 1
        self._episode_reward += reward
        self._total_reward += reward
        done = raw_obs.last()
        info = self.get_info() if done else {}
        # each step will set the dictionary to emtpy
        return obs, reward, done, info

    def take_action(self, obs, action):

        if action == 0:
            action_mapped = actions.RAW_FUNCTIONS.no_op()
        elif action == 1:
            action_mapped = self.train_scv(obs)
        elif action == 2:
            action_mapped = self.harvest_minerals(obs)
        elif action == 3:
            action_mapped = self.build_supply_depot(obs)
        elif action == 4:
            action_mapped = self.build_refinery(obs)
        elif action == 5:
            action_mapped = self.build_command_center(obs)
        elif action == 6:
            action_mapped = self.harvest_gas(obs)

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
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
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
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if obs.observation.player.minerals >= 100:
            x = random.randint(0, 64)
            y = random.randint(0, 64)
            supply_depot_xy = (x, y)
            distances = self.get_distances(obs, scvs, supply_depot_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
                "now", scv.tag, supply_depot_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def train_scv(self, obs):
        completed_command_center = self.get_my_completed_units_by_type(
            obs, units.Terran.CommandCenter)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(completed_command_center) > 0 and obs.observation.player.minerals >= 50
                and free_supply > 0):
            for cc in range(len(completed_command_center)):
                command_center = self.get_my_units_by_type(obs, units.Terran.Barracks)[cc]
                if command_center.order_length < 1:
                    return actions.RAW_FUNCTIONS.Train_Marine_quick("now", command_center.tag)
            x = random.randint(0, len(completed_command_center))
            command_center = self.get_my_units_by_type(obs, units.Terran.CommandCenter)[x]
            if command_center.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_SCV_quick("now", command_center.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_refinery(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        geysers = [unit for unit in obs.observation.raw_units
                   if unit.unit_type in [
                       units.Neutral.ProtossVespeneGeyser,
                       units.Neutral.PurifierVespeneGeyser,
                       units.Neutral.RichVespeneGeyser,
                       units.Neutral.ShakurasVespeneGeyser,
                       units.Neutral.VespeneGeyser,
                   ]]
        scv = random.choice(scvs)
        distances = self.get_distances(obs, geysers, (scv.x, scv.y))
        geyser = geysers[np.argmin(distances)]
        return actions.RAW_FUNCTIONS.Build_Refinery_pt(
                "now", scv.tag, (geyser.x, geyser.y))

    def harvest_gas(self, obs):
        scvs = self.get_my_completed_units_by_type(obs, units.Terran.SCV)
        refineries = self.get_my_completed_units_by_type(obs, units.Terran.Refinery)
        if len(refineries) > 0:
            refinery = random.choice(refineries)
            if refinery.features.FeatureUnit.ideal_harvesters > refinery.features.FeatureUnit.assigned_harvesters:
                scv = random.choice(scvs)
                return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                    "now", scv.tag, refinery.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_command_center(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        completed_command_center = self.get_my_completed_units_by_type(
            obs, units.Terran.CommandCenter)
        cc_location = [completed_command_center[0].x, completed_command_center[0].y]
        if len(completed_command_center) == 1 and obs.observation.player.minerals >= 400:
            command_center_xy = [completed_command_center[0].x + 5, completed_command_center[0].y]
            # distances = self.get_distances(obs, scvs, command_center_xy)
            scv = random.choice(scvs)
            return actions.RAW_FUNCTIONS.Build_CommandCenter_pt(
                "now", scv.tag, command_center_xy)
        else:
            return actions.RAW_FUNCTIONS.no_op()

    def close(self):

        if self.env is not None:
            self.env.close()
        super().close()

    def render(self, mode='human', close=False):
        pass
