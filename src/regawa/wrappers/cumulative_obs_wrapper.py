from typing import Any, SupportsFloat

import gymnasium as gym
import logging
from regawa import GroundObs
from regawa.model.base_grounded_model import TemporalGroundObs

logger = logging.getLogger(__name__)


def merge_obs[T: GroundObs | TemporalGroundObs](obs: T, prev_obs: T) -> T:
    return prev_obs | obs  # type: ignore


class CumulativeObsWrapper[T: GroundObs | TemporalGroundObs](
    gym.Wrapper[T, GroundObs, T, GroundObs]
):
    """A wrapper that accumulates observations over time. Existing keys are updated, new keys are added."""

    def __init__(
        self,
        env: gym.Env[T, GroundObs],
    ) -> None:
        super().__init__(env)
        self.prev_obs: T | None = None

    def step(
        self,
        action: GroundObs,
    ) -> tuple[
        T,
        SupportsFloat,
        bool,
        bool,
        dict[str, Any],
    ]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        new_obs = merge_obs(obs, self.prev_obs or {})

        self.prev_obs = new_obs

        return new_obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[T, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed)

        self.prev_obs = obs

        return obs, info
