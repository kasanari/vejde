from typing import Any, TypeVar
from pyRDDLGym.core.compiler.model import RDDLPlanningModel  # type: ignore
import gymnasium as gym
from gymnasium import spaces
from pyRDDLGym import RDDLEnv
from functools import cache

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
WrapperObsType = spaces.Dict
WrapperActType = spaces.Tuple


class RDDLAddNonFluents(gym.Wrapper[WrapperActType, WrapperObsType, ObsType, ActType]):
    def __init__(self, env: RDDLEnv, only_add_on_reset: bool = False) -> None:
        super().__init__(env)
        self.only_add_on_reset = only_add_on_reset

    @property
    @cache
    def _non_fluent_values(self) -> dict[str, int]:
        # model = self.model
        # return dict(
        #     model.ground_vars_with_values(model.non_fluents)  # type: ignore
        # )

        non_fluents = (
            self.env.model.ast.non_fluents.init_non_fluent
            if hasattr(self.env.model.ast.non_fluents, "init_non_fluent")
            else []
        )
        nf_vals = {
            RDDLPlanningModel.ground_var(name, params): value
            for (name, params), value in non_fluents
        }

        return nf_vals

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
        obs, reward, terminated, truncated, info = self.env.step(actions)

        if not self.only_add_on_reset:
            obs |= self._non_fluent_values

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed)

        obs |= self._non_fluent_values

        return obs, info
