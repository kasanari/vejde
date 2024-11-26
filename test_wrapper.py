from wrappers.wrapper import GroundedRDDLGraphWrapper, register_env
from wrappers.kg_wrapper import KGRDDLGraphWrapper
import numpy as np
import gymnasium as gym


def policy(obs):
    nonzero = np.flatnonzero(obs["edge_attr"])
    if len(nonzero) == 0:
        return [0, 0]

    obj = next(iter(nonzero))

    obj = obs["edge_index"][obj][0]

    button = next(
        p for p, c in obs["edge_index"] if c == obj and not ((p, c) == (obj, obj))
    )

    return [1, button]


def test_grounded(seed):
    # domain = "Elevators_MDP_ippc2011"
    # domain = "conditional_bandit.rddl"
    # instance = "conditional_bandit_i0.rddl"
    domain = "Elevators_POMDP_ippc2011"
    instance = 1
    env_id = register_env()
    env = gym.make(
        env_id,
        domain=domain,
        instance=instance,
        # add_inverse_relations=False,
        # types_instead_of_objects=False,
        render_mode="idx",
        pomdp=True,
    )

    # domain = "RecSim_ippc2023"
    # domain = "SkillTeaching_MDP_ippc2011"
    # env = GroundedRDDLGraphWrapper(domain, instance)
    obs, info = env.reset(seed=seed)
    # env.render()
    done = False
    time = 0
    sum_reward = 0
    while not done:
        time += 1
        action = env.action_space.sample()
        # action = [1, 3]
        # action = policy(obs)
        # print(info["state"].edge_attributes)
        # print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        # print(reward)
        # print(obs["edge_attr"])
        # env.render()
        # exit()
        # print(env.last_action)
        done = terminated or truncated

        sum_reward += reward
        # print(obs)
        # print(action)
        # print(reward)
    return sum_reward


if __name__ == "__main__":
    return_ = [test_grounded(i) for i in range(1)]
    print(np.mean(return_))
