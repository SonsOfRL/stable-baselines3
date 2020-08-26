import gym
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from gym import spaces
import logging
import numpy as np
import random

logger = logging.getLogger(__name__)


class CMGEnv(gym.Env):
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

        for i, sd in enumerate(supply_depot):
            self.SCVs.append(sd)
            obs[i] = np.array([sd.x, sd.y, sd[2]])

        for i, cc in enumerate(command_center):
            self.SCVs.append(cc)
            obs[i] = np.array([cc.x, cc.y, cc[2]])

        for i, r in enumerate(refinery):
            self.SCVs.append(r)
            obs[i] = np.array([r.x, r.y, r[2]])
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
            derived_action = np.floor((action - 1) / 8)    #TODO WE DO NOT NEED IDX IN GATHERING ENVIRONMENTS
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
        supply_depots = self.get_units_by_type(obs, units.Terran.SupplyDepot)
        scvs = self.get_units_by_type(obs, units.Terran.SCV)
        if (len(supply_depots) == 0 and obs.observation.player.minerals >= 100 and
                len(scvs) > 0):
            supply_depot_xy = (22, 26)  #TODO THIS COORDINATE IS RANDOMLY SET, GIVE RANDOM LOCATION TO BUILD
            distances = self.get_distances(obs, scvs, supply_depot_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
                "now", scv.tag, supply_depot_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def close(self):

        if self.env is not None:
            self.env.close()
        super().close()

    def render(self, mode='human', close=False):
        pass
