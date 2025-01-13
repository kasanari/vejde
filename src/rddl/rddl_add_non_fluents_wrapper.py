from typing import Any, TypeVar
from pyRDDLGym.core.compiler.model import RDDLPlanningModel  # type: ignore
import gymnasium as gym
from gymnasium import spaces
from pyRDDLGym import RDDLEnv
from functools import cache

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


class RDDLAddNonFluents(gym.Wrapper[spaces.Tuple, spaces.Dict, ObsType, ActType]):


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

        nf_vals = {}

        non_fluents = self.env.model.ast.non_fluents.init_non_fluent  # type: ignore
        for (name, params), value in non_fluents:
            gname = RDDLPlanningModel.ground_var(name, params)
            nf_vals[gname] = value

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
        self, seed: int | None = None
    ) -> tuple[
        dict[str, bool | None],
        dict[str, Any],
    ]:
        obs, info = self.env.reset(seed=seed)

        obs |= self._non_fluent_values

        return obs, info
