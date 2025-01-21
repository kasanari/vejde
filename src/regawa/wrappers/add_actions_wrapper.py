from typing import Any, TypeVar

from regawa.model.base_model import BaseModel
import gymnasium as gym
import numpy as np

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


def add_actions_to_obs(
    obs: dict[str, Any], actions: dict[str, bool | None]
) -> dict[str, Any]:
    obs_with_actions = {a: v for a, v in actions.items()} | obs
    return obs_with_actions


class AddActionWrapper(
    gym.Wrapper[gym.spaces.Tuple, gym.spaces.Dict, gym.spaces.Tuple, gym.spaces.Dict]
):
    """
    Adds actions to the previous observation.
    """

    def __init__(self, env: gym.Env[gym.spaces.Tuple, gym.spaces.Dict]) -> None:
        super().__init__(env)
        self.env = env

    @staticmethod
    def _add_actions_to_obs(
        model: BaseModel, obs: dict[str, Any], actions: dict[str, int]
    ) -> dict[str, bool | None]:
        action_groundings = model.action_groundings  # type: ignore

        boolean_actions: dict[str, np.bool_] = {
            k: np.bool_(v) for k, v in actions.items()
        }

        new_actions: dict[str, bool | None] = {
            k: boolean_actions.get(k, None) for k in action_groundings if k not in obs
        }

        obs_with_actions = add_actions_to_obs(obs, new_actions)
        return obs_with_actions

    @staticmethod
    def _dynamic_add_actions_to_obs(
        obs: dict[str, Any], actions: dict[str, int]
    ) -> dict[str, bool]:
        boolean_actions = {k: np.bool_(v) for k, v in actions.items()}
        obs_with_actions = add_actions_to_obs(obs, boolean_actions)
        return obs_with_actions

    def step(
        self,
        actions: gym.spaces.Dict,
    ) -> tuple[
        tuple[dict[str, bool | None], dict[str, bool | None]],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        (last_obs, next_obs), reward, terminated, truncated, info = self.env.step(
            actions
        )

        obs_with_actions = self._dynamic_add_actions_to_obs(last_obs, actions)

        return (obs_with_actions, next_obs), reward, terminated, truncated, info

    def reset(
        self, seed: int | None = None
    ) -> tuple[
        dict[str, bool | None],
        dict[str, Any],
    ]:
        (last_obs, obs), info = self.env.reset(seed=seed)

        obs_with_actions = self._dynamic_add_actions_to_obs(last_obs, {})

        o = (obs_with_actions, obs)

        return o, info
