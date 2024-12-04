import copy
from typing import Any, TypeVar

import gymnasium as gym
from gymnasium import spaces


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


class LastObsWrapper(gym.Wrapper[spaces.Tuple, spaces.Dict, ObsType, ActType]):
    """
    A wrapper that adds the last observation to the current observation.
    """

    def __init__(self, env: gym.Env[ObsType, ActType]) -> None:
        super().__init__(env)
        self.observation_space = spaces.Tuple(
            (self.env.observation_space, self.env.observation_space)
        )
        self.last_obs: dict[str, Any] = {}

    def step(
        self,
        actions: ActType,
    ) -> tuple[
        tuple[dict[str, bool | None], dict[str, bool | None]],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        next_obs, reward, terminated, truncated, info = self.env.step(actions)

        obs = (copy.deepcopy(self.last_obs), copy.deepcopy(next_obs))

        self.last_obs = next_obs

        return obs, reward, terminated, truncated, info

    def reset(
        self, seed: int | None = None
    ) -> tuple[
        dict[str, bool | None],
        dict[str, Any],
    ]:
        next_obs, info = self.env.reset(seed=seed)

        obs = (copy.deepcopy(self.last_obs), copy.deepcopy(next_obs))

        self.last_obs = next_obs

        return obs, info
