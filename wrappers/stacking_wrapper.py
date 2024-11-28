from pprint import pprint
from typing import Any

import pyRDDLGym

from wrappers.utils import get_groundings


def stack_obs(
    horizon: int,
    obs: dict[str, Any],
    buffer: list[dict[str, Any]],
    observed_keys: list[str],
) -> dict[str, list[Any]]:
    result: dict[str, list[bool | None]] = {key: [] for key in observed_keys}

    lengths = {}

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
    observed_keys: set[str],
    buffer: list[dict[str, Any]],
    horizon: int,
):
    obs = {k: bool(v) for k, v in obs.items()}
    observed_keys = set(obs.keys()) | observed_keys
    stacked_obs, lengths = stack_obs(horizon, obs, buffer, observed_keys)
    return stacked_obs, lengths, observed_keys


class StackingWrapper:
    def __init__(self, env: pyRDDLGym.RDDLEnv) -> None:
        self.env = env  # ActionInObsWrapper(env)
        self.buffer: list[dict[str, Any]] = []
        self.observed_keys: set[str] = set()
        self.iteration = 0
        self.horizon = int(env.horizon)

    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[str, str], float, bool, bool, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed)
        o, lengths, observed_keys = create_obs(obs, set(), [], self.horizon)

        self.observed_keys = observed_keys
        self.buffer = []
        self.iteration = 0

        return (o, lengths), info

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

        observed_keys: set[str] = (
            set(next_obs.keys())
            | self.observed_keys  # | set(obs_with_actions.keys()) |
        )

        o, lengths, observed_keys = create_obs(
            next_obs, observed_keys, self.buffer, self.horizon
        )

        self.observed_keys = observed_keys
        self.buffer = self.buffer + [next_obs]

        return (o, lengths), reward, terminated, truncated, info

    @property
    def unwrapped(self):
        return self.env.unwrapped


if __name__ == "__main__":
    env = pyRDDLGym.make("Tamarisk_POMDP_ippc2014", 1)
    env = StackingWrapper(env)
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step({})
        done = terminated or truncated
    # print(obs, reward, terminated, truncated, info)
    pprint(obs)
