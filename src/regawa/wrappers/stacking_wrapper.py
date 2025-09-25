from collections import deque
from typing import Any, SupportsFloat

import gymnasium as gym

from regawa.model.base_grounded_model import (
    GroundObs,
    Grounding,
    GroundingRange,
    StackedGroundObs,
)


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
    obs: GroundObs,
    buffer: dict[Grounding, deque[Any]],
) -> StackedGroundObs:
    for key in obs:
        if key not in buffer:
            buffer[key] = deque()
        buffer[key].append(obs[key])

    return buffer


class StackingWrapper(gym.Wrapper[StackedGroundObs, GroundObs, GroundObs, GroundObs]):
    def __init__(self, env: gym.Env[GroundObs, GroundObs]) -> None:
        self.env = env  # ActionInObsWrapper(env)
        self.buffer: dict[Grounding, deque[GroundingRange]] = {}
        self.observed_keys: set[str] = set()
        self.iteration = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[StackedGroundObs, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed)
        o = create_obs(obs, {})

        self.buffer = o  # type: ignore
        self.iteration = 0

        new_obs = {k: list(v) for k, v in o.items()}

        return new_obs, info  # type: ignore

    def step(
        self,
        action: GroundObs,
    ) -> tuple[StackedGroundObs, SupportsFloat, bool, bool, dict[str, Any]]:
        """""
        Stacks observations
        obs= {
            "key1" [t1, t2, t3, t4],
            "key2" [t1, t2, t3, t4],
            "key3" [None, None, t3, t4],
        }
        """ ""

        next_obs, reward, terminated, truncated, info = self.env.step(action)

        o = create_obs(next_obs, self.buffer)

        self.buffer = o  # type: ignore

        new_obs = {k: list(v) for k, v in o.items()}

        return new_obs, reward, terminated, truncated, info  # type: ignore
