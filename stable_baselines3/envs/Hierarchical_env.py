from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from gym import spaces
import logging
import numpy as np
from stable_baselines3.envs.base_env import SC2Env
from stable_baselines3.common.custom_space import TreeDiscreteDict
import random

logger = logging.getLogger(__name__)


class HierarchEnv(SC2Env):
    metadata = {'render.modes': ['human']}
    default_settings = {
        'map_name': "Simple64",
        'players': [sc2_env.Agent(sc2_env.Race.terran),
                    sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.easy)],
        'agent_interface_format': features.AgentInterfaceFormat(
            action_space=actions.ActionSpace.RAW,
            use_raw_units=True,
            raw_resolution=64),

        'realtime': False,
        "disable_fog": True
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.env = None
        self._num_step = 0
        self._episode_reward = 0
        self._episode = 0
        self.obs = None
        self.Hierarch = False

        self.refinery_counter1 = 0
        self.refinery_counter2 = 0

        # 0 no operation
        #
        self.action_space = TreeDiscreteDict({
            0: {
                "parent": None,
                "childs": [1, 2, 3],
                "action_dim": 3

            },

            1: {
                "parent": 0,
                "childs": None,
                "action_dim": 17

            },
            2: {
                "parent": 0,
                "childs": None,
                "action_dim": 17

            },

            3: {
                "parent": 0,
                "childs": None,
                "action_dim": 17

            }


        })
        # [0: x, 1: y, 2: hp]
        self.observation_space = spaces.Box(
            low=0,
            high=100000,
            shape=(23 * 1,),
            dtype=np.uint64
        )

    def reset(self):
        if self.env is None:
            self.init_env()

        raw_obs = self.env.reset()[0]

        self._episode += 1
        self._num_step = 0
        self._episode_reward = 0
        self.new_game()

        return self.get_state(raw_obs)

    def init_env(self):
        args = {**self.default_settings, **self.kwargs}
        self.env = sc2_env.SC2Env(**args)

    def step(self, action):

        raw_obs = self.take_action(action)
        reward = raw_obs.reward
        obs = self.get_state(raw_obs)
        self._num_step += 1
        self._episode_reward += reward
        self._total_reward += reward
        done = raw_obs.last()
        info = self.get_info() if done else {}
        # each step will set the dictionary to empty
        return obs, reward, done, info

    def take_action(self, action):

        if action == 0:
            action_mapped = actions.RAW_FUNCTIONS.no_op()
        elif action == 1:
            action_mapped = self.all_attack()
        elif action == 2:
            action_mapped = self.attack_base_marine()
        elif action == 3:
            action_mapped = self.attack_base_marauder()
        elif action == 4:
            action_mapped = self.attack_random_enemy()
        elif action == 5:
            action_mapped = self.all_attack_exp()
        elif action == 6:
            action_mapped = self.harvest_minerals()
        elif action == 7:
            action_mapped = self.train_scv()
        elif action == 8:
            action_mapped = self.build_supply_depot()
        elif action == 9:
            action_mapped = self.build_barracks()
        elif action == 10:
            action_mapped = self.build_refinery()
        elif action == 11:
            action_mapped = self.harvest_gas()
        elif action == 12:
            action_mapped = self.attach_reactor()
        elif action == 13:
            action_mapped = self.attach_techlab()
        elif action == 14:
            action_mapped = self.train_marauder()
        elif action == 15:
            action_mapped = self.protect_base()
        else:
            action_mapped = self.train_marine()

        raw_obs = self.env.step([action_mapped])[0]
        return raw_obs

    def get_state(self, obs):
        if obs.first():
            command_center = self.get_my_units_by_type(
                obs, units.Terran.CommandCenter)[0]
            self.base_top_left = (command_center.x < 32)
        self.obs = obs

        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        marauders = self.get_my_units_by_type(obs, units.Terran.Marauder)

        minerals = obs.observation.player.minerals
        vespene = obs.observation.player.vespene
        refineries = self.get_my_units_by_type(obs, units.Terran.Refinery)

        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if free_supply < 0:
            free_supply = 0
        if obs.observation.player.minerals >= 100:
            can_afford_supply_depot = 1
        else:
            can_afford_supply_depot = 0

        if obs.observation.player.minerals >= 150:
            can_afford_barracks = 1
        else:
            can_afford_barracks = 0

        if obs.observation.player.minerals >= 75:
            can_afford_refinery = 1
        else:
            can_afford_refinery = 0

        if obs.observation.player.minerals >= 50:
            can_afford_marine = 1
        else:
            can_afford_marine = 0

        enemy_scvs = self.get_enemy_units_by_type(obs, units.Terran.SCV)
        enemy_idle_scvs = [scv for scv in enemy_scvs if scv.order_length == 0]
        enemy_command_centers = self.get_enemy_units_by_type(
            obs, units.Terran.CommandCenter)
        enemy_supply_depots = self.get_enemy_units_by_type(
            obs, units.Terran.SupplyDepot)
        enemy_completed_supply_depots = self.get_enemy_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        enemy_barrackses = self.get_enemy_units_by_type(obs, units.Terran.Barracks)
        enemy_completed_barrackses = self.get_enemy_completed_units_by_type(
            obs, units.Terran.Barracks)
        enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)
        enemy_marauders = self.get_enemy_units_by_type(obs, units.Terran.Marauder)


        obs = np.zeros((23, 1), dtype=np.int64)

        obs[0] = len(scvs)
        obs[1] = len(supply_depots)
        obs[2] = len(command_centers)
        obs[3] = len(refineries)
        obs[4] = can_afford_barracks
        obs[5] = can_afford_supply_depot
        obs[6] = can_afford_marine
        obs[7] = can_afford_refinery
        obs[8] = minerals
        obs[9] = free_supply
        obs[10] = len(idle_scvs)
        obs[11] = len(completed_supply_depots)
        obs[12] = len(completed_barrackses)
        obs[13] = len(marauders)
        obs[14] = 0
        obs[15] = len(enemy_command_centers)
        obs[16] = len(enemy_scvs)
        obs[17] = len(enemy_supply_depots)
        obs[18] = len(enemy_completed_supply_depots)
        obs[19] = len(enemy_barrackses)
        obs[20] = len(enemy_completed_barrackses)
        obs[21] = len(enemy_marauders)
        obs[22] = len(enemy_marines)

        return obs.reshape(-1)

    def get_my_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.ENEMY]

    def get_random_enemy_unit(self, obs):
        return [unit for unit in obs.observation.raw_units
                if unit.alliance == features.PlayerRelative.ENEMY]

    def get_my_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.ENEMY]

    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def train_scv(self):
        scvs = self.get_my_completed_units_by_type(self.obs, units.Terran.SCV)
        completed_command_center = self.get_my_completed_units_by_type(
            self.obs, units.Terran.CommandCenter)
        free_supply = (self.obs.observation.player.food_cap -
                       self.obs.observation.player.food_used)
        if (len(completed_command_center) > 0 and self.obs.observation.player.minerals >= 50
                and free_supply > 0):
            command_center = self.get_my_completed_units_by_type(self.obs, units.Terran.CommandCenter)[0]
            if command_center.order_length < 2:
                return actions.RAW_FUNCTIONS.Train_SCV_quick("now", command_center.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def harvest_minerals(self):
        scvs = self.get_my_units_by_type(self.obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        if len(idle_scvs) > 0:
            mineral_patches = [unit for unit in self.obs.observation.raw_units
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
            distances = self.get_distances(self.obs, mineral_patches, (scv.x, scv.y))
            mineral_patch = mineral_patches[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                "now", scv.tag, mineral_patch.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def new_game(self):
        self.base_top_left = None
        self.previous_state = None
        self.previous_action = None

    def build_supply_depot(self):
        supply_depots = self.get_my_units_by_type(self.obs, units.Terran.SupplyDepot)
        scvs = self.get_my_units_by_type(self.obs, units.Terran.SCV)
        if (len(supply_depots) < 10 and self.obs.observation.player.minerals >= 100 and
                len(scvs) > 0):
            randomx = random.randint(-3, 3)
            randomy = random.randint(-3, 3)
            supply_depot_xy = (20 + randomx, 26 + randomy) if self.base_top_left else (37 + randomx, 42 + randomy)
            distances = self.get_distances(self.obs, scvs, supply_depot_xy)
            scv = self.select_scv()
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
                "now", scv.tag, supply_depot_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_barracks(self):
        completed_supply_depots = self.get_my_completed_units_by_type(
            self.obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(self.obs, units.Terran.Barracks)
        scvs = self.get_my_units_by_type(self.obs, units.Terran.SCV)
        if (len(completed_supply_depots) > 0 and len(barrackses) < 5 and
                self.obs.observation.player.minerals >= 150 and len(scvs) > 0):
            randomx = random.randint(-3, 3)
            randomy = random.randint(-3, 3)
            barracks_xy = (23 + randomx, 21 + randomy) if self.base_top_left else (34 + randomx, 46 + randomy)
            distances = self.get_distances(self.obs, scvs, barracks_xy)
            spy = actions.RAW_FUNCTIONS
            scv = self.select_scv()
            return actions.RAW_FUNCTIONS.Build_Barracks_pt(
                "now", scv.tag, barracks_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def train_marine(self):
        completed_barrackses = self.get_my_completed_units_by_type(
            self.obs, units.Terran.Barracks)
        free_supply = (self.obs.observation.player.food_cap -
                       self.obs.observation.player.food_used)
        if (len(completed_barrackses) > 0 and self.obs.observation.player.minerals >= 50
                and free_supply > 0):
            barracks = self.get_my_units_by_type(self.obs, units.Terran.Barracks)[0]
            if barracks.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def attach_techlab(self):
        barracks = self.get_my_completed_units_by_type(self.obs, units.Terran.Barracks)
        if len(
                barracks) > 0 and self.obs.observation.player.minerals >= 100 and self.obs.observation.player.vespene > 50:
            barrack = random.choice(barracks)
            return actions.RAW_FUNCTIONS.Build_TechLab_Barracks_quick("now", barrack.tag)
        return actions.RAW_FUNCTIONS.no_op()  # TODO optimize attachments-- destroy randomness

    def attach_reactor(self):
        barracks = self.get_my_completed_units_by_type(self.obs, units.Terran.Barracks)
        if len(
                barracks) > 0 and self.obs.observation.player.minerals >= 100 and self.obs.observation.player.vespene > 75:
            barrack = random.choice(barracks)
            return actions.RAW_FUNCTIONS.Build_Reactor_Barracks_quick("now", barrack.tag)
        return actions.RAW_FUNCTIONS.no_op()  # TODO optimize attachments-- destroy randomness

    def build_refinery(self):
        ccs = self.get_my_completed_units_by_type(self.obs, units.Terran.CommandCenter)
        refineries = self.get_my_units_by_type(self.obs, units.Terran.Refinery)
        geysers = [unit for unit in self.obs.observation.raw_units
                   if unit.unit_type in [
                       units.Neutral.ProtossVespeneGeyser,
                       units.Neutral.PurifierVespeneGeyser,
                       units.Neutral.RichVespeneGeyser,
                       units.Neutral.ShakurasVespeneGeyser,
                       units.Neutral.VespeneGeyser,
                   ]]
        scvs = self.get_my_completed_units_by_type(self.obs, units.Terran.SCV)

        if len(geysers) > 0 and len(refineries) < 2 and len(ccs) > 0 and len(scvs) > 0:
            scv = self.select_scv()
            command_center_xy = [ccs[0].x, ccs[0].y]
            distances = self.get_distances(self.obs, geysers, command_center_xy)
            if len(refineries) == 0:
                geyser = geysers[np.argmin(distances)]
                return actions.RAW_FUNCTIONS.Build_Refinery_pt("now", scv.tag, geyser.tag)

            else:
                k = len(refineries)
                geyser = geysers[np.argpartition(distances, k)[k]]
                return actions.RAW_FUNCTIONS.Build_Refinery_pt("now", scv.tag, geyser.tag)

        return actions.RAW_FUNCTIONS.no_op()

    def harvest_gas(self):
        scvs = self.get_my_completed_units_by_type(self.obs, units.Terran.SCV)
        refineries = self.get_my_completed_units_by_type(self.obs, units.Terran.Refinery)
        refinery_tags = []
        new_refineries = []
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]

        if len(refineries) > 0 and self.refinery_counter1 < 2 and self.refinery_counter2 < 2:
            if len(refineries) == 1 and self.refinery_counter1 >= 2:
                return actions.RAW_FUNCTIONS.no_op()
            else:
                for refinery in range(len(refineries)):
                    if refineries[refinery].tag not in refinery_tags:
                        refinery_tags.append(refineries[refinery].tag)
                        new_refineries.append(refineries[refinery])

                if len(new_refineries) > 0 and len(scvs) > 0:

                    choice = random.randint(0, len(new_refineries) - 1)
                    if new_refineries[choice] is not None:

                        if choice == 0 and self.refinery_counter1 < 3 and len(idle_scvs) > 0:
                            scv = random.choice(idle_scvs)
                            self.refinery_counter1 = self.refinery_counter1 + 1
                            return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                                "now", scv.tag, new_refineries[choice].tag)

                        elif choice == 1 and self.refinery_counter2 < 3 and len(idle_scvs) > 0:
                            scv = random.choice(idle_scvs)
                            self.refinery_counter2 = self.refinery_counter2 + 1

                            return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                                "now", scv.tag, new_refineries[choice].tag)

        return actions.RAW_FUNCTIONS.no_op()

    def train_marauder(self):
        completed_barrackses = self.get_my_completed_units_by_type(
            self.obs, units.Terran.Barracks)
        free_supply = (self.obs.observation.player.food_cap -
                       self.obs.observation.player.food_used)
        if (len(completed_barrackses) > 0 and self.obs.observation.player.minerals >= 50
                and free_supply > 0):
            barracks = self.get_my_units_by_type(self.obs, units.Terran.Barracks)[0]
            if barracks.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Marauder_quick("now", barracks.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def select_scv(self):
        scvs = self.get_my_completed_units_by_type(self.obs, units.Terran.SCV)
        mineral_patches = [unit for unit in self.obs.observation.raw_units
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
        ccs = self.get_my_completed_units_by_type(self.obs, units.Terran.CommandCenter)
        if len(ccs) > 0 and len(scvs) > 0:
            command_center = ccs[0]
            cc_distance = self.get_distances(self.obs, mineral_patches, (command_center.x, command_center.y))
            mineral = mineral_patches[np.argmin(cc_distance)]
            distances = self.get_distances(self.obs, scvs, (mineral.x, mineral.y))
            scv = scvs[np.argmin(distances)]
            return scv
        else:
            return random.choice(scvs)

    def protect_base(self):
        army_tags = []
        marines = self.get_my_units_by_type(self.obs, units.Terran.Marine)
        marauders = self.get_enemy_units_by_type(self.obs, units.Terran.Marauder)
        if len(marines) > 0:
            for marine in range(len(marines)):
                army_tags.append(marines[marine].tag)

        if len(marauders) > 0:
            for marauder in range(len(marauders)):
                army_tags.append(marauders[marauder].tag)

        if len(marauders) > 0 or len(marines) > 0:
            attack_xy = (28, 23) if self.base_top_left else (29, 43)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", army_tags, (attack_xy[0], attack_xy[1]))

        return actions.RAW_FUNCTIONS.no_op()

    def attack_base_marauder(self):
        marauders = self.get_my_units_by_type(self.obs, units.Terran.Marauder)

        if len(marauders) > 0:
            attack_xy = (38, 44) if self.base_top_left else (19, 23)
            distances = self.get_distances(self.obs, marauders, attack_xy)
            marauder = marauders[np.argmax(distances)]
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", marauder.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def attack_base_marine(self):
        marines = self.get_my_units_by_type(self.obs, units.Terran.Marine)

        if len(marines) > 0:
            attack_xy = (38, 44) if self.base_top_left else (19, 23)
            distances = self.get_distances(self.obs, marines, attack_xy)
            marine = marines[np.argmax(distances)]
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", marine.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def all_attack(self):
        army_tags = []
        marines = self.get_my_units_by_type(self.obs, units.Terran.Marine)
        marauders = self.get_enemy_units_by_type(self.obs, units.Terran.Marauder)
        if len(marines) > 0:
            for marine in range(len(marines)):
                army_tags.append(marines[marine].tag)

        if len(marauders) > 0:
            for marauder in range(len(marauders)):
                army_tags.append(marauders[marauder].tag)

        if len(marauders) > 0 or len(marines) > 0:
            attack_xy = (38, 44) if self.base_top_left else (19, 23)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", army_tags, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def attack_random_enemy(self):
        my_marines = self.get_my_units_by_type(self.obs, units.Terran.Marine)
        enemy_marines = self.get_enemy_units_by_type(self.obs, units.Terran.Marine)
        enemies = self.get_random_enemy_unit(self.obs)

        if len(my_marines) > len(enemy_marines) + 20 and len(enemies) > 0:
            enemy = random.choice(enemies)
            marine = random.choice(my_marines)
            return actions.RAW_FUNCTIONS.Attack_unit(
                "now", marine.tag, enemy.tag)

        return actions.RAW_FUNCTIONS.no_op()

    def all_attack_exp(self):
        army_tags = []
        marines = self.get_my_units_by_type(self.obs, units.Terran.Marine)
        marauders = self.get_enemy_units_by_type(self.obs, units.Terran.Marauder)
        if len(marines) > 0:
            for marine in range(len(marines)):
                army_tags.append(marines[marine].tag)

        if len(marauders) > 0:
            for marauder in range(len(marauders)):
                army_tags.append(marauders[marauder].tag)

        if len(marauders) > 0 or len(marines) > 0:
            attack_xy = (15, 44) if self.base_top_left else (35, 23)
            x_offset = random.randint(-6, 6)
            y_offset = random.randint(-6, 6)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", army_tags, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def close(self):

        if self.env is not None:
            self.env.close()
        super().close()

    def render(self, mode='human', close=False):
        pass
