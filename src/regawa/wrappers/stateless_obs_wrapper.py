from typing import Any, SupportsFloat, TypeVar, Callable

import gymnasium as gym
import numpy as np

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


def create_stateless_wrapper(
    transform: Callable[[dict[Any, Any]], dict[Any, Any]],
) -> Callable[
    [gym.Env[gym.spaces.Dict, gym.spaces.Dict]],
    gym.Wrapper[gym.spaces.Dict, gym.spaces.Dict, gym.spaces.Dict, gym.spaces.Dict],
]:
    """
    Creates a stateless observation wrapper that applies a transformation to the observations.
    """

    def wrapper(env: gym.Env[gym.spaces.Dict, gym.spaces.Dict]) -> StateLessWrapper:
        return StateLessWrapper(env, transform)

    return wrapper


class StateLessWrapper(
    gym.Wrapper[gym.spaces.Dict, gym.spaces.Dict, gym.spaces.Dict, gym.spaces.Dict]
):
    """
    Stateless observation wrapper class that
    """

    def __init__(
        self,
        env: gym.Env[gym.spaces.Dict, gym.spaces.Dict],
        transform: Callable[[dict[Any, Any]], dict[Any, Any]],
    ) -> None:
        super().__init__(env)
        self.env = env
        self.transform = transform

    def step(
        self,
        action: gym.spaces.Dict,
    ) -> tuple[
        dict[str, bool | None],
        SupportsFloat,
        bool,
        bool,
        dict[str, Any],
    ]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.transform(obs), reward, terminated, truncated, info

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[
        dict[str, bool | None],
        dict[str, Any],
    ]:
        obs, info = self.env.reset(seed=seed)
        return self.transform(obs), info
