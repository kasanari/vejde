import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
from kg_wrapper import KGRDDLGraphWrapper
import numpy as np


class IndexedActionWrapper:
    def __init__(self, env: KGRDDLGraphWrapper):
        self.env = env
        self.action_space = MultiDiscrete([env.num_objects, env.num_actions])

        self.action_mask = np.stack(
            [np.array([x for x in v.values()]) for v in env.action_mask.values()]
        )

        pass

    def step(self, action):
        object_id = self.env.idx_to_obj[action[0]]
        action_fluent = self.env.action_fluents[action[1]]
        return self.env.step({f"{action_fluent}___{object_id}": 1})

    def reset(self):
        o, i = self.env.reset()
        return self.translate_obs(o)


if __name__ == "__main__":
    domain = "Elevators_MDP_ippc2011"
    instance = 1
    env = KGRDDLGraphWrapper(domain, instance, enforce_action_constraints=False)
    env = IndexedActionWrapper(env)
    obs = env.reset()
    print(obs)
    action = env.action_space.sample()
    action_fluent = action[1]
    valid_objects = np.flatnonzero(env.action_mask[action_fluent])
    object_idx = np.random.choice(valid_objects)
    new_action = (object_idx, action_fluent)
    print(new_action)
    obs, reward, term, trunc, info = env.step(new_action)
    print(obs, reward, done, info)
    env.close()
