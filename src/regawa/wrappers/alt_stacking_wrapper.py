from collections.abc import Sequence
from typing import Any, NamedTuple, SupportsFloat
from itertools import groupby
import gymnasium as gym

from regawa.model.base_grounded_model import (
    GroundObs,
    GroundingRange,
    StackedGroundObs,
    TemporalGroundObs,
)


class TimeEntry(NamedTuple):
    start_time: int
    end_time: int


class RelativeTimeEntry(NamedTuple):
    duration: int
    distance_to_prev: int


class CompressedStack(NamedTuple):
    te: TimeEntry
    value: GroundingRange


class RelativeCompressedStack(NamedTuple):
    rte: RelativeTimeEntry
    value: GroundingRange


def get_time_entry(
    current_time: int,
    current_value: GroundingRange,
    prev_time: int,
    prev_value: GroundingRange,
    start_time: int,
) -> tuple[CompressedStack, CompressedStack | None]:
    time_skip = current_time != prev_time + 1  # check for gaps in the sequence
    value_change = current_value != prev_value  # check for changes in value

    if time_skip:
        # if there is a gap, end the previous segment and start a new one
        return (
            CompressedStack(
                TimeEntry(start_time, prev_time),
                prev_value,
            ),  # end previous segment
            CompressedStack(
                TimeEntry(current_time, current_time), current_value
            ),  # start new segment
        )

    if value_change:
        # if the value has changed, end the previous segment and start a new one
        return (
            CompressedStack(
                TimeEntry(start_time, current_time), prev_value
            ),  # end previous segment
            CompressedStack(
                TimeEntry(current_time, current_time),
                current_value,
            ),  # start new segment
        )

    # if the value has not changed, or there are any gaps, continue current segment
    return CompressedStack(
        TimeEntry(start_time, current_time),
        current_value,
    ), None


def compress_stack(o: Sequence[tuple[int, GroundingRange]]):
    """
    [(time1: int, grounding_range1: GroundingRange), (time2: int, grounding_range2: GroundingRange), ...]
    ->
    [(start_time1: int, end_time1: int, grounding_range1: GroundingRange), (start_time2: int, end_time2: GroundingRange), ...]
    """

    compressed_stack: list[tuple[TimeEntry, GroundingRange]] = []

    first_entry = o[0]

    segment = CompressedStack(
        TimeEntry(first_entry[0], first_entry[0]),
        first_entry[1],
    )

    for time, value in o[1:]:
        segment, new_segment = get_time_entry(
            time,
            value,
            segment.te.end_time,
            segment.value,
            segment.te.start_time,
        )

        if new_segment is not None:
            compressed_stack.append(segment)
            segment = new_segment

    compressed_stack.append(segment)

    return tuple(compressed_stack)


def to_relative_time_entry(te: TimeEntry, prev_te: TimeEntry) -> RelativeTimeEntry:
    duration = te.end_time - te.start_time + 1
    distance_to_prev = max(te.start_time - prev_te.end_time, 0)
    return RelativeTimeEntry(duration, distance_to_prev)


def stack_to_relative_time(
    compressed_stack: Sequence[CompressedStack],
) -> list[RelativeCompressedStack]:
    relative_stack: list[RelativeCompressedStack] = []

    first_entry = compressed_stack[0]
    prev_te = first_entry.te

    for cs in compressed_stack:
        rte = to_relative_time_entry(cs.te, prev_te)
        relative_stack.append(RelativeCompressedStack(rte, cs.value))
        prev_te = cs.te

    return relative_stack


def obs_to_relative_time(
    obs: StackedGroundObs,
):
    return {
        grounding: stack_to_relative_time(compressed_stack)
        for grounding, compressed_stack in obs.items()
    }


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

    new_obs: StackedGroundObs = {
        grounding: tuple([(time, value) for (time, _), value in values])
        for grounding, values in groupby(sorted_obs, key=lambda x: x[0][1])
    }

    new_obs = compress_stacked_obs(new_obs)

    relative = obs_to_relative_time(new_obs)

    return relative


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
