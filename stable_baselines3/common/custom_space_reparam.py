import numpy as np
from gym.spaces import Discrete


class ReparamHieDict(Discrete):
    """
    """

    def __init__(self, action_dict):

        self.action_dict = action_dict
        n = action_dict["levels"][-1] * action_dict["action_dim"]
        super().__init__(n)