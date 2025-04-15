from typing import Any, TypeVar

import gymnasium as gym
import logging
from regawa.model import GroundValue

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
WrapperObsType = dict[GroundValue, Any]
WrapperActType = dict[GroundValue, Any]

logger = logging.getLogger(__name__)


def check_if_equal(obs1: WrapperObsType, obs2: WrapperObsType) -> bool:
    equal_size = len(obs1) == len(obs2)
    if not equal_size:
        return False

    # Check if all keys are the same
    keys1 = set(obs1.keys())
    keys2 = set(obs2.keys())
    # diff = keys1.symmetric_difference(keys2)
    # logger.debug(f"Keys in obs1 but not in obs2: {diff}")
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
        discount: float = 1.0,
    ) -> None:
        super().__init__(env)
        self.last_obs = None
        self.discount = discount

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

        skipped_steps = 0
        while check_if_equal(obs, self.last_obs):
            skipped_steps += 1
            obs, reward, terminated, truncated, info = self.env.step({})
            accumulated_reward += (self.discount**skipped_steps) * reward
            if terminated or truncated:
                logger.debug(f"Terminated or truncated after {skipped_steps} steps")
                break

        self.last_obs = obs
        if skipped_steps > 1:
            logger.debug(f"Skipped steps: {skipped_steps}")

        return obs, accumulated_reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed)

        self.last_obs = obs

        return obs, info
