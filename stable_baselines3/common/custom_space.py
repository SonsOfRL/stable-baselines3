import numpy as np
from gym.spaces import Discrete


class TreeDiscreteDict(Discrete):
    """
    """

    def __init__(self, action_dict):

        self.action_dict = action_dict
        n = sum(node["action_dim"] for node in action_dict.values() if node["childs"] is None)
        super().__init__(n)