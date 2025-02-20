from collections import deque
from copy import deepcopy
from pprint import pprint
from typing import Any, TypeVar

import gymnasium as gym

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


def stack_obs(
    horizon: int,
    obs: dict[str, Any],
    buffer: list[dict[str, Any]],
    observed_keys: set[str],
) -> tuple[dict[str, list[Any]], dict[str, int]]:
    result: dict[str, list[bool | None]] = {key: [] for key in observed_keys}

    lengths: dict[str, int] = {}

    for step, o in enumerate(buffer):
        o = buffer[step]
        for key in observed_keys:
            if key in o:
                result[key].append(o[key])
                lengths[key] = len(result[key])

    if len(buffer) < horizon:
        for key in observed_keys:
            if key in obs:
                result[key].append(obs[key])
                lengths[key] = len(result[key])

        # Fill in the rest of the buffer with None
        for k in result:
            if len(result[k]) < horizon:
                result[k] += [None] * (horizon - len(result[k]))

    for k, v in result.items():
        assert len(v) == horizon

    return result, lengths


def create_obs(
    obs: dict[str, Any],
    buffer: dict[str, deque[Any]],
):
    obs.update({k: bool(v) for k, v in obs.items() if isinstance(v, bool)})
    obs = {k: v for k, v in obs.items() if v is not None}

    for key in obs:
        if key not in buffer:
            buffer[key] = deque()
        buffer[key].append(obs[key])

    return buffer


class StackingWrapper(gym.Wrapper):
    def __init__(self, env) -> None:
        self.env = env  # ActionInObsWrapper(env)
        self.buffer: dict[str, deque[Any]] = {}
        self.observed_keys: set[str] = set()
        self.iteration = 0
        # self.horizon = int(env.horizon)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, str], float, bool, bool, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        obs, info = self.env.reset(seed=seed)
        o = create_obs(obs, {})

        # self.observed_keys = observed_keys
        self.buffer = deepcopy(o)
        self.iteration = 0

        new_obs = {k: list(v) for k, v in o.items()}

        return new_obs, info  # type: ignore

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[dict[str, str], float, bool, dict[str, Any]]:
        """""
        Stacks observations
        obs= {
            "key1" [t1, t2, t3, t4],
            "key2" [t1, t2, t3, t4],
            "key3" [None, None, t3, t4],
        }
        """ ""
        # (obs_with_actions, next_obs), reward, terminated, truncated, info = (
        #     self.env.step(actions)
        # )

        next_obs, reward, terminated, truncated, info = self.env.step(actions)

        o = create_obs(next_obs, self.buffer)

        # self.observed_keys = observed_keys
        self.buffer = deepcopy(o)

        new_obs = {k: list(v) for k, v in o.items()}

        return new_obs, reward, terminated, truncated, info  # type: ignore


if __name__ == "__main__":
    env = pyRDDLGym.make("Tamarisk_POMDP_ippc2014", 1)  # type: ignore
    env = StackingWrapper(env)
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step({})  # type: ignore
        done = terminated or truncated
    # print(obs, reward, terminated, truncated, info)
    pprint(obs)  # type: ignore
