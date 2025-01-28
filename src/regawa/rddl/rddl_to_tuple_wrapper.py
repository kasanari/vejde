from typing import Any, TypeVar
import gymnasium as gym
from gymnasium import spaces
from pyRDDLGym import RDDLEnv
from functools import cache
from .rddl_utils import rddl_ground_to_tuple

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
WrapperObsType = spaces.Dict
WrapperActType = spaces.Tuple


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

        obs = {rddl_ground_to_tuple(k): v for k, v in obs.items()}

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed)

        obs = {rddl_ground_to_tuple(k): v for k, v in obs.items()}

        return obs, info
