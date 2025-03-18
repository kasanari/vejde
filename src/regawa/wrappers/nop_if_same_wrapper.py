from typing import Any, TypeVar

import gymnasium as gym

from regawa.model import GroundValue

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
WrapperObsType = dict[GroundValue, Any]
WrapperActType = dict[GroundValue, Any]


def check_if_equal(obs1: WrapperObsType, obs2: WrapperObsType) -> bool:
    equal_size = len(obs1) == len(obs2)
    if not equal_size:
        return False

    # Check if all keys are the same
    keys1 = set(obs1.keys())
    keys2 = set(obs2.keys())
    equal_keys = keys1 == keys2
    if not equal_keys:
        return False

    # Check if all values are the same
    for key in keys1:
        if obs1[key] != obs2[key]:
            return False

    return True


class NoOpIfSameWrapper(gym.Wrapper[WrapperActType, WrapperObsType, ObsType, ActType]):
    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
    ) -> None:
        super().__init__(env)
        self.last_obs = None

    def step(
        self,
        actions: ActType,
    ) -> tuple[
        dict[GroundValue, Any],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        obs, reward, terminated, truncated, info = self.env.step(actions)

        accumulated_reward = reward

        while check_if_equal(obs, self.last_obs):
            obs, reward, terminated, truncated, info = self.env.step({})
            accumulated_reward += reward
            if terminated or truncated:
                break

        self.last_obs = obs

        return obs, accumulated_reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed)

        self.last_obs = obs

        return obs, info
