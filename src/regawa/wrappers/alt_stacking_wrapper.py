from collections.abc import Sequence
from typing import Any, SupportsFloat
from itertools import groupby
import gymnasium as gym

from regawa.model.base_grounded_model import (
    GroundObs,
    GroundingRange,
    StackedGroundObs,
    TemporalGroundObs,
)


def compress_stack(o: Sequence[tuple[int, GroundingRange]]):
    """
    [(time1: int, grounding_range1: GroundingRange), (time2: int, grounding_range2: GroundingRange), ...]
    ->
    [(start_time1: int, end_time1: int, grounding_range1: GroundingRange), (start_time2: int, end_time2: GroundingRange), ...]
    """

    compressed_stack: list[tuple[tuple[int, int], GroundingRange]] = []

    start_time, prev_time, prev_value = o[0][0], o[0][0], o[0][1]

    for time, value in o[1:]:
        time_skip = time != prev_time + 1
        value_skip = value != prev_value

        if time_skip:
            compressed_stack.append(
                ((start_time, prev_time), prev_value)
            )  # add the previous segment
            start_time = time  # start a new segment
            prev_value = value

        if value_skip:
            compressed_stack.append(((start_time, time), prev_value))
            start_time = time  # start a new segment
            prev_value = value

        prev_time = time

    compressed_stack.append(((start_time, prev_time), prev_value))

    return compressed_stack


def compress_stacked_obs(
    obs: StackedGroundObs,
) -> StackedGroundObs:
    """
    {
    grounding1: str, [(time1: int, value1: Any), (time2: int, value2: Any), ...]),
    grounding2: str, [(time1: int, value1: Any), (time2: int, value1: Any), ...]),
    }
    ->
    grounding1: str, [(start_time1: int, end_time1: int, value1: Any), (start_time2: int, end_time2: int, value2: Any), ...]),
    grounding2: str, [(start_time1: int, end_time1: Any), ...]),
    """

    new_obs: StackedGroundObs = {
        grounding: compress_stack(values) for grounding, values in obs.items()
    }

    return new_obs


def create_obs(obs: TemporalGroundObs) -> StackedGroundObs:
    sorted_obs = sorted(obs.items(), key=lambda x: x[0][1])
    grouped_obs = groupby(sorted_obs, key=lambda x: x[0][1])

    new_obs: StackedGroundObs = {
        grounding: tuple([(time, value) for (time, _), value in values])
        for grounding, values in grouped_obs
    }

    new_obs = compress_stacked_obs(new_obs)

    return new_obs


class StackingWrapper(gym.Wrapper[StackedGroundObs, GroundObs, GroundObs, GroundObs]):
    def __init__(self, env: gym.Env[GroundObs, GroundObs]) -> None:
        self.env = env  # ActionInObsWrapper(env)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[StackedGroundObs, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed)
        o = create_obs(obs)

        return o, info  # type: ignore

    def step(
        self,
        action: GroundObs,
    ) -> tuple[StackedGroundObs, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        o = create_obs(obs)

        return o, reward, terminated, truncated, info  # type: ignore
