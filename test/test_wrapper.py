import gymnasium as gym
import numpy as np

from wrappers.add_actions_wrapper import AddActionWrapper
from wrappers.labelwrapper import LabelingWrapper
from wrappers.last_obs_wrapper import LastObsWrapper
from wrappers.stacking_last_obs_wrapper import LastObsStackingWrapper
from wrappers.stacking_wrapper import StackingWrapper
from rddl import register_env


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


def save_dot(dot, path):
    with open(path, "w") as f:
        f.write(dot)


def test_render(seed):
    domain = "rddl/conditional_bandit.rddl"
    instance = "rddl/conditional_bandit_i0.rddl"
    # domain = "Elevators_MDP_ippc2011"
    # instance = 1

    env_id = register_env()
    env = gym.make(
        env_id,
        domain=domain,
        instance=instance,
    )

    # domain = "RecSim_ippc2023"
    # domain = "SkillTeaching_MDP_ippc2011"
    # env = GroundedRDDLGraphWrapper(domain, instance)
    obs, info = env.reset(seed=seed)
    dot = env.render()
    save_dot(dot, "render/0.dot")
    done = False
    time = 0
    sum_reward = 0
    while not done:
        time += 1

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        dot = env.render()
        save_dot(dot, f"render/{time}.dot")
        done = terminated or truncated
        sum_reward += reward

    return sum_reward


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


def test_stacking_wrappers():
    import pyRDDLGym

    domain = "rddl/conditional_bandit.rddl"
    instance = "rddl/conditional_bandit_i0.rddl"
    # env_id = register_env()
    # env = gym.make(
    #     env_id,
    #     domain=domain,
    #     instance=instance,
    #     render_mode="idx",
    # )

    env: gym.Env[gym.spaces.Dict, gym.spaces.Dict] = pyRDDLGym.make(
        domain, instance, enforce_action_constraints=True
    )  # type: ignore

    env = LastObsStackingWrapper(AddActionWrapper(LastObsWrapper(env)))

    obs, info = env.reset()

    action1 = {"press___red": 1, "press___green": 0}
    obs, reward, terminated, truncated, info = env.step(action1)

    action2 = {"press___red": 0, "press___green": 1}
    obs, reward, terminated, truncated, info = env.step(action2)
    pass


def test_wrappers():
    import pyRDDLGym

    domain = "rddl/conditional_bandit.rddl"
    instance = "rddl/conditional_bandit_i0.rddl"
    # env_id = register_env()
    # env = gym.make(
    #     env_id,
    #     domain=domain,
    #     instance=instance,
    #     render_mode="idx",
    # )

    env: gym.Env = pyRDDLGym.make(domain, instance, enforce_action_constraints=True)  # type: ignore

    env = LabelingWrapper(AddActionWrapper(LastObsWrapper(env)))

    obs, info = env.reset()

    action1 = {"press___red": 1, "press___green": 0}
    obs, reward, terminated, truncated, info = env.step(action1)

    action2 = {"press___red": 0, "press___green": 1}
    obs, reward, terminated, truncated, info = env.step(action2)
    pass


if __name__ == "__main__":
    # return_ = [test_grounded(i) for i in range(1)]
    # print(np.mean(return_))
    # test_wrappers()
    test_render(1)
