from collections.abc import Callable
from typing import Any
import gymnasium as gym
import numpy as np
import pytest

from wrappers.add_actions_wrapper import AddActionWrapper
from wrappers.labelwrapper import LabelingWrapper
from wrappers.last_obs_wrapper import LastObsWrapper
from wrappers.stacking_last_obs_wrapper import LastObsStackingWrapper
from rddl import register_env, register_pomdp_env
from gymnasium.utils.env_checker import check_env

env_id = register_env()
pomdp_env_id = register_pomdp_env()


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


def step_with_render(
    env: gym.Env,
    obs: dict[str, Any],
    sum_reward: int,
    time: int,
    done: bool,
    policy: Callable[[dict[str, Any]], Any],
):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    dot = env.render()
    save_dot(dot, f"render/{time}.dot")
    done = terminated or truncated
    sum_reward += reward
    time += 1
    return sum_reward, time, done


def step(
    env: gym.Env,
    obs: dict[str, Any],
    sum_reward: int,
    time: int,
    done: bool,
    policy: Callable[[dict[str, Any]], Any],
):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    sum_reward += reward
    time += 1
    return obs, sum_reward, time, done


@pytest.mark.parametrize("env_id", [env_id, pomdp_env_id])
def test_render(seed, env_id):
    # domain = "rddl/conditional_bandit.rddl"
    # instance = "rddl/conditional_bandit_i0.rddl"
    domain = "Elevators_MDP_ippc2011"
    instance = 1
    # domain = "SysAdmin_MDP_ippc2011"
    # instance = 1

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
    save_dot(dot, f"render/{domain}/0.dot")
    done = False
    time = 0
    sum_reward = 0

    while not done:
        sum_reward, time, done = step_with_render(env, sum_reward, time, done)

    return sum_reward


def check_obs_in_space(key: str, obs: dict[str, Any], obs_space: gym.spaces.Space):
    if isinstance(obs_space, gym.spaces.Dict):
        for k, v in obs.items():
            check_obs_in_space(f"{key}.{k}", v, obs_space.spaces[k])
        return True
    if isinstance(obs_space, gym.spaces.Sequence):
        assert obs in obs_space, f"{key} not in {obs_space}"
        return True
    if isinstance(obs_space, gym.spaces.Discrete):
        assert obs in obs_space, f"{key} not in {obs_space}"
        return True

    assert False


def test_pomdp_wrapper():
    # domain = "Elevators_MDP_ippc2011"
    # domain = "conditional_bandit.rddl"
    # instance = "conditional_bandit_i0.rddl"
    # domain = "Elevators_POMDP_ippc2011"
    # instance = 1

    seed = 1
    domain = "rddl/counting_bandit.rddl"
    instance = "rddl/counting_bandit_i1.rddl"
    env_id = register_pomdp_env()
    env = gym.make(
        env_id,
        domain=domain,
        instance=instance,
    )

    # domain = "RecSim_ippc2023"
    # domain = "SkillTeaching_MDP_ippc2011"
    # env = GroundedRDDLGraphWrapper(domain, instance)
    obs, info = env.reset(seed=seed)

    check_obs_in_space("", obs, env.observation_space)
    check_env(env)
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


@pytest.mark.parametrize("env_id", [env_id, pomdp_env_id])
def test_rddl_domains(env_id):
    from rddlrepository import RDDLRepoManager

    manager = RDDLRepoManager(rebuild=True)
    problems = manager.list_problems()

    import logging

    logger = logging.getLogger("wrappers")
    logger.setLevel(logging.ERROR)

    problems_to_skip = [
        "ComplexSysAdmin_rddlsim",  # RDDLSim error
        "Tamarisk_POMDP_ippc2014",  # encoding error
        "Tamarisk_MDP_ippc2014",  # encoding error
        "TriangleTireworld_POMDP_ippc2014",  # encoding error
        "TriangleTireworld_MDP_ippc2014",  # encoding error
    ]

    seed = 0

    for domain in problems:
        p = manager.get_problem(domain)
        if domain in problems_to_skip:
            continue

        try:
            env = gym.make(
                env_id,
                domain=domain,
                instance=p.instances[0],
            )
        except Exception as e:
            print(f"Error in {domain}: {e}")
            assert False, f"Error initing in {domain}: {e}"

        obs, info = env.reset(seed=seed)

        done = False

        check_obs_in_space("", obs, env.observation_space)

        try:
            check_env(env.unwrapped)
        except Exception as e:
            print(f"Error in {domain}: {e}")
            assert False, f"Error checking in {domain}: {e}"

        policy = lambda _: env.action_space.sample()

        time = 0
        sum_reward = 0

        while not done:
            obs, sum_reward, time, done = step(env, obs, sum_reward, time, done, policy)


def test_wrapper():
    seed = 1
    domain = "EarthObservation_ippc2018"  # "rddl/conditional_bandit.rddl"
    instance = 1
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

    check_obs_in_space("", obs, env.observation_space)
    check_env(env.unwrapped)

    done = False
    time = 0
    sum_reward = 0
    policy = lambda _: env.action_space.sample()
    while not done:
        obs, sum_reward, time, done = step(env, obs, sum_reward, time, done, policy)

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


def test_last_obs_wrappers():
    import pyRDDLGym

    domain = "rddl/conditional_bandit.rddl"
    instance = "rddl/conditional_bandit_i0.rddl"

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
    # test_last_obs_wrappers()
    # test_render(1)
    # test_wrapper()
    test_wrapper()
