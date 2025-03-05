from typing import Any, TypeVar

import gymnasium as gym

from regawa.model import GroundValue
from regawa.model.base_grounded_model import BaseGroundedModel

ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')
WrapperObsType = dict[GroundValue, Any]
WrapperActType = dict[GroundValue, Any]


def add_constants(ground_model: BaseGroundedModel):
    constant_vals = {
        g: ground_model.constant_value(g)
        for g in ground_model.constant_groundings
    }

    def f(obs: dict[GroundValue, Any]) -> dict[GroundValue, Any]:
        return obs | constant_vals

    return f


class AddConstantsWrapper(
    gym.Wrapper[WrapperActType, WrapperObsType, ObsType, ActType]
):
    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        ground_model: BaseGroundedModel,
        only_add_on_reset: bool = False,
    ) -> None:
        super().__init__(env)
        self.only_add_on_reset = only_add_on_reset
        self.transform = add_constants(ground_model)

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

        if not self.only_add_on_reset:
            obs = self.transform(obs)

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed)

        obs = self.transform(obs)

        return obs, info
