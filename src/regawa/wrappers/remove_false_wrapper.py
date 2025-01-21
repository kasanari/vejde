from typing import Any, TypeVar, SupportsFloat
import gymnasium as gym

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


def remove_false(obs: dict[str, Any]) -> dict[str, Any]:
    obs_with_actions = {
        a: v for a, v in obs.items() if v is not False
    }
    return obs_with_actions


class RemoveFalseWrapper(
    gym.Wrapper[gym.spaces.Dict, gym.spaces.Dict, gym.spaces.Dict, gym.spaces.Dict]
):
    """
    Adds actions to the previous observation.
    """

    def __init__(self, env: gym.Env[gym.spaces.Dict, gym.spaces.Dict]) -> None:
        super().__init__(env)
        self.env = env

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
        filtered_obs = remove_false(obs)
        return filtered_obs, reward, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None
    ) -> tuple[
        dict[str, bool | None],
        dict[str, Any],
    ]:
        obs, info = self.env.reset(seed=seed)
        filtered_obs = remove_false(obs)
        return filtered_obs, info
