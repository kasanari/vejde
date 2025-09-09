from collections.abc import Callable
from typing import Any, SupportsFloat, TypeVar

import gymnasium as gym

from regawa import GroundObs

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


def create_stateless_wrapper(
    transform: Callable[[GroundObs], GroundObs],
) -> Callable[
    [gym.Env[GroundObs, GroundObs]],
    gym.Wrapper[GroundObs, GroundObs, GroundObs, GroundObs],
]:
    """
    Creates a stateless observation wrapper that applies a transformation to the observations.
    """

    def wrapper(env: gym.Env[GroundObs, GroundObs]) -> StateLessWrapper:
        return StateLessWrapper(env, transform)

    return wrapper


class StateLessWrapper(
    gym.Wrapper[GroundObs, GroundObs, GroundObs, GroundObs]
):
    """
    Stateless observation wrapper class that
    """

    def __init__(
        self,
        env: gym.Env[GroundObs, GroundObs],
        transform: Callable[[GroundObs], GroundObs],
    ) -> None:
        super().__init__(env)
        self.env = env
        self.transform = transform

    def step(
        self,
        action: GroundObs,
    ) -> tuple[
        GroundObs,
        SupportsFloat,
        bool,
        bool,
        dict[str, Any],
    ]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.transform(obs), reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[
        GroundObs,
        dict[str, Any],
    ]:
        obs, info = self.env.reset(seed=seed)
        return self.transform(obs), info
