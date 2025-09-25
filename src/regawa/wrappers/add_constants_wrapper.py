from typing import Any, SupportsFloat

import gymnasium as gym

from regawa import GroundObs
from regawa.model import BaseGroundedModel


def add_constants_fn(ground_model: BaseGroundedModel):
    constant_vals = {
        g: ground_model.constant_value(g) for g in ground_model.constant_groundings
    }

    def f(obs: GroundObs) -> GroundObs:
        return obs | constant_vals

    return f


class AddConstantsWrapper(gym.Wrapper[GroundObs, GroundObs, GroundObs, GroundObs]):
    """
    Adds constant values to the observation, if there are constants defined in a grounded model.
    When `only_add_on_reset` is True, constants are only added in the first step.
    """

    def __init__(
        self,
        env: gym.Env[GroundObs, GroundObs],
        ground_model: BaseGroundedModel,
        only_add_on_reset: bool = False,
    ) -> None:
        super().__init__(env)
        self.only_add_on_reset = only_add_on_reset
        self.transform = add_constants_fn(ground_model)

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

        if not self.only_add_on_reset:
            obs = self.transform(obs)

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[GroundObs, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed)

        obs = self.transform(obs)

        return obs, info
