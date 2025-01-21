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


@cache
def split_rddl_grounding(grounding: str):
    pred, *args = grounding.split("___")
    args = args[0].split("__") if args else ()
    return (pred, *args)


@cache
def merge_rddl_grounding(grounding: tuple[str, ...]):
    action_fluent, *params = grounding
    return f"{action_fluent}___{'__'.join(params)}" if params else action_fluent


class RDDLToTuple(gym.Wrapper[WrapperActType, WrapperObsType, ObsType, ActType]):
    def __init__(self, env: RDDLEnv) -> None:
        super().__init__(env)

    def step(
        self,
        actions: ActType,
    ) -> tuple[
        dict[str, bool | None],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        actions = {merge_rddl_grounding(k): v for k, v in actions.items()}

        obs, reward, terminated, truncated, info = self.env.step(actions)

        obs = {split_rddl_grounding(k): v for k, v in obs.items()}

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed)

        obs = {split_rddl_grounding(k): v for k, v in obs.items()}

        return obs, info
