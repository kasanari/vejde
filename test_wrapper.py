from wrappers.wrapper import GroundedRDDLGraphWrapper, register_env
from wrappers.kg_wrapper import KGRDDLGraphWrapper
import numpy as np
import gymnasium as gym


def counting_policy(state):
    if np.array(state["light___r_m"], dtype=bool).sum() > 3:
        return [1, 3]

    if np.array(state["light___g_m"], dtype=bool).sum() > 3:
        return [1, 1]

    return [0, 0]


def policy(state):
    if state["enough_light___r_m"]:
        return [1, 3]

    if state["enough_light___g_m"]:
        return [1, 1]

    return [0, 0]


def test_grounded(seed):
    # domain = "Elevators_MDP_ippc2011"
    # domain = "conditional_bandit.rddl"
    # instance = "conditional_bandit_i0.rddl"
    # domain = "Elevators_POMDP_ippc2011"
    # instance = 1

    domain = "rddl/counting_bandit.rddl"
    instance = "rddl/counting_bandit_i1.rddl"
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
        # action = env.action_space.sample()
        # action = [1, 3]
        print(info["rddl_state"])

        action = policy(info["rddl_state"])
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
