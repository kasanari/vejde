from collections import deque
from typing import Any, SupportsFloat
from itertools import groupby
import gymnasium as gym

from regawa.model.base_grounded_model import (
    GroundObs,
    StackedGroundObs,
    TemporalGroundObs,
)


def create_obs(
    obs: TemporalGroundObs
) -> StackedGroundObs:
    
    sorted_obs = sorted(
        obs.items(), key=lambda x: x[0][1]
    )
    grouped_obs = groupby(sorted_obs, key=lambda x: x[0][1])

    new_obs: StackedGroundObs = {
        grounding: tuple([
            (time, value)
            for (time, _), value in values
        ])
        for grounding, values in grouped_obs
    }
    
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
