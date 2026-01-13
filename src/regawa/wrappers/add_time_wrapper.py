import logging
from typing import Any, SupportsFloat

import gymnasium as gym

from regawa.model import GroundObs, TemporalGroundObs

logger = logging.getLogger(__name__)


def add_time_to_obs(obs: GroundObs, time: int) -> TemporalGroundObs:
    return {(time, key): value for key, value in obs.items()}


class AddTimeWrapper(gym.Wrapper[TemporalGroundObs, GroundObs, GroundObs, GroundObs]):
    """A wrapper that adds the current time step to the observation keys."""

    def __init__(
        self,
        env: gym.Env[GroundObs, GroundObs],
    ) -> None:
        super().__init__(env)
        self.iteration = 0

    def step(
        self,
        action: GroundObs,
    ) -> tuple[
        TemporalGroundObs,
        SupportsFloat,
        bool,
        bool,
        dict[str, Any],
    ]:
        iteration = self.iteration + 1
        obs, reward, terminated, truncated, info = self.env.step(action)

        new_obs = add_time_to_obs(obs, iteration)

        info["time"] = iteration

        self.iteration = iteration
        return new_obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[TemporalGroundObs, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed)

        new_obs = add_time_to_obs(obs, 0)

        self.iteration = 0

        return new_obs, info
