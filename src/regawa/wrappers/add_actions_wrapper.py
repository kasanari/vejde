from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np

from regawa.model import GroundObs
from regawa.model.base_grounded_model import BaseGroundedModel


def add_actions_to_obs(obs: GroundObs, actions: GroundObs) -> GroundObs:
    obs_with_actions = {a: v for a, v in actions.items()} | obs
    return obs_with_actions


def fn_add_actions_to_obs(grounded_model: BaseGroundedModel):
    action_groundings = grounded_model.action_groundings  # type: ignore

    def add_actions_to_obs(obs: GroundObs, actions: GroundObs) -> GroundObs:
        boolean_actions = {k: np.bool_(v) for k, v in actions.items()}

        new_actions = {
            k: boolean_actions.get(k, None) for k in action_groundings if k not in obs
        }

        obs_with_actions = add_actions_to_obs(obs, new_actions)
        return obs_with_actions

    return add_actions_to_obs


def dynamic_add_actions_to_obs(obs: GroundObs, actions: GroundObs) -> GroundObs:
    boolean_actions = {k: np.bool_(v) for k, v in actions.items()}
    obs_with_actions = add_actions_to_obs(obs, boolean_actions)
    return obs_with_actions


class AddActionWrapper(gym.Wrapper[GroundObs, GroundObs, GroundObs, GroundObs]):
    """
    Adds the previous action to the observation. Only the most recent action is set to true.
    """

    def __init__(
        self,
        env: gym.Env[GroundObs, GroundObs],
        grounded_model: BaseGroundedModel | None = None,
    ) -> None:
        super().__init__(env)
        self.env = env

        self.add_action_func = (
            fn_add_actions_to_obs(grounded_model)
            if grounded_model
            else dynamic_add_actions_to_obs
        )

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

        obs_with_actions = self.add_action_func(obs, action)

        return obs_with_actions, reward, terminated, truncated, info

    # Listening to: "Mawarukagami" by "Perfume"
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[
        GroundObs,
        dict[str, Any],
    ]:
        obs, info = self.env.reset(seed=seed)

        obs_with_actions = self.add_action_func(obs, {})

        return obs_with_actions, info
