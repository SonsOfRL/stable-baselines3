import gym
from pysc2.env import sc2_env
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from gym import spaces
import logging
import numpy as np
import random

logger = logging.getLogger(__name__)


class CMGEnv(base_agent.BaseAgent):
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
        self.supply_depot = []
        self.command_center = []
        self.refinery = []

        # 0 no operation
        # 1~32 move
        # 33~122 attack
        self.action_space = spaces.Discrete(123)
        # [0: x, 1: y, 2: hp]
        self.observation_space = spaces.Box(
            low=0,
            high=64,
            shape=(19, 3),
            dtype=np.uint8
        )

    def reset(self):
        if self.env is None:
            self.init_env()

        self.SCVs = []
        self.supply_depot = []
        self.command_center = []
        self.refinery = []

        raw_obs = self.env.reset()[0]
        return self.get_derived_obs(raw_obs)

    def init_env(self):
        args = {**self.default_settings, **self.kwargs}
        self.env = sc2_env.SC2Env(**args)

    def get_derived_obs(self, raw_obs):
        obs = np.zeros((19, 3), dtype=np.uint8)
        SCVs = self.get_units_by_type(raw_obs, units.Terran.SCV, 1)
        supply_depot = self.get_units_by_type(raw_obs, units.Terran.SupplyDepot, 1)
        command_center = self.get_units_by_type(raw_obs, units.Terran.CommandCenter, 1)
        refinery = self.get_units_by_type(raw_obs, units.Terran.Refinery, 1)

        self.SCVs = []
        self.supply_depot = []
        self.command_center = []
        self.refinery = []

        for i, m in enumerate(SCVs):
            self.SCVs.append(m)
            obs[i] = np.array([m.x, m.y, m[2]])
        k = len(obs)

        for i, sd in enumerate(supply_depot):
            self.SCVs.append(sd)
            obs[i+k] = np.array([sd.x, sd.y, sd[2]])
        k = len(obs)

        for i, cc in enumerate(command_center):
            self.SCVs.append(cc)
            obs[i+k] = np.array([cc.x, cc.y, cc[2]])
        k = len(obs)

        for i, r in enumerate(refinery):
            self.SCVs.append(r)
            obs[i+k] = np.array([r.x, r.y, r[2]])
        return obs

    def step(self, action):
        raw_obs = self.take_action(action)
        reward = raw_obs.reward
        obs = self.get_derived_obs(raw_obs)
        # each step will set the dictionary to emtpy
        return obs, reward, raw_obs.last(), {}

    def take_action(self, action):
        if action <= 64:
            action_mapped = actions.RAW_FUNCTIONS.no_op()
        else:
            derived_action = np.floor((action - 1) / 8)    #TODO TAKE ACTION WILL CHANGE
            idx = (action - 1) % 8
            if derived_action == 0:
                action_mapped = self.move_up(idx)
            elif derived_action == 1:
                action_mapped = self.move_down(idx)
            elif derived_action == 2:
                action_mapped = self.move_left(idx)
            else:
                action_mapped = self.move_right(idx)

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
        scvs = self.get_units_by_type(obs, units.Terran.SCV)
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
        scvs = self.get_units_by_type(obs, units.Terran.SCV)
        refineries = self.get_units_by_type(obs, units.Terran.Refinery)
        if len(refineries) > 0:
            refinery = random.choice(refineries)
            if features.FeatureUnit.ideal_harvesters > features.FeatureUnit.assigned_harvesters:  #TODO ideal harvesters for gas or mineral ?
                scv = random.choice(scvs)
                return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                    "now", scv.tag, refinery.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_command_center(self, obs):
        scvs = self.get_units_by_type(obs, units.Terran.SCV)
        completed_command_center = self.get_my_completed_units_by_type(
            obs, units.Terran.CommandCenter)
        if len(completed_command_center) == 1 and obs.observation.player.minerals >= 400:
            command_center_xy = (34, 32)                                   #TODO not sure about the coordinates
            distances = self.get_distances(obs, scvs, command_center_xy)
            scv = scvs[np.argmin(distances)]
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
