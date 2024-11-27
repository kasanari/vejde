from pprint import pprint
from typing import Any

import pyRDDLGym


class StackingWrapper:
    def __init__(self, env: pyRDDLGym) -> None:
        self.env = env
        self.buffer: list[dict[str, Any]] = []
        self.observed_keys = set()
        self.iteration = 0
        self.horizon = env.horizon

    def create_obs(
        self, buffer: list[dict[str, Any]], observed_keys: list[str]
    ) -> dict[str, list[Any]]:
        result = {key: [None] * (self.horizon + 1) for key in observed_keys}

        lengths = {}

        for step, o in enumerate(buffer):
            o = buffer[step]
            for key in observed_keys:
                if key in o:
                    result[key][step] = o[key]
                    lengths[key] = step + 1

        return result, lengths

    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[str, str], float, bool, bool, dict[str, Any]]:
        self.buffer = []
        self.iteration = 0
        self.observed_keys = set()
        obs, info = self.env.reset(seed=seed)

        self.buffer.append(obs)
        self.observed_keys |= set(obs.keys())

        stacked_obs = self.create_obs(self.buffer, self.observed_keys)

        return stacked_obs, info

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

        obs, reward, terminated, truncated, info = self.env.step(actions)

        self.buffer.append(obs)
        self.observed_keys |= set(obs.keys())

        o = self.create_obs(self.buffer, self.observed_keys)

        return o, reward, terminated, truncated, info


if __name__ == "__main__":
    env = pyRDDLGym.make("Tamarisk_POMDP_ippc2014", 1)
    env = StackingWrapper(env)
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step({})
        done = terminated or truncated
    # print(obs, reward, terminated, truncated, info)
    pprint(obs)
