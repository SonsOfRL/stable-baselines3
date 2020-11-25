from stable_baselines3.common.distributions import TreeCategoricalDistrubution as tcd
import torch

action_dict = {
    0: {
        "parent": None,
        "childs": [1, 2],
        "action_dim": 2
    },
    1: {
        "parent": 0,
        "childs": None,
        "action_dim": 3
    },
    2: {
        "parent": 0,
        "childs": [3, 4],
        "action_dim": 2
    },
    3: {
        "parent": 2,
        "childs": None,
        "action_dim": 4
    },
    4: {
        "parent": 2,
        "childs": None,
        "action_dim": 5
    }


}
action_logits = torch.cat([torch.rand(node["action_dim"]) for node in action_dict.values()]).unsqueeze(0).repeat(20, 1)

dist = tcd(action_dict)
dist.proba_distribution(action_logits)
sample = dist.sample()
print(dist.log_prob(sample))
print(action_dict)
print(action_dict[0]["action_dim"])


