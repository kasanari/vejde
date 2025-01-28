from itertools import chain
from typing import Any, TypeVar
import gymnasium as gym
from gymnasium import spaces
from pyRDDLGym import RDDLEnv
from functools import cache

from .rddl_model import RDDLModel
from regawa.wrappers.utils import predicate

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
WrapperObsType = spaces.Dict
WrapperActType = spaces.Tuple


@cache
def convert_enum_key(key: str, value: str) -> str:
    return f"{key}^^^{value}"


class RDDLConvertEnums(gym.Wrapper[WrapperActType, WrapperObsType, ObsType, ActType]):
    def __init__(self, env: RDDLEnv, only_add_on_reset: bool = False) -> None:
        super().__init__(env)

        self.rddl_model = RDDLModel(env.unwrapped.model)

        self.enum_fluents = list(chain(*self.rddl_model.enum_fluents.values()))
        self.enum_values = self.rddl_model.enum_values

    def transform_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        new_data = {
            convert_enum_key(k, v): True
            for k, v in obs.items()
            if k in self.enum_fluents and v is not None
        }

        obs = {**obs, **new_data}

        obs = {k: v for k, v in obs.items() if predicate(k) not in self.enum_fluents}
        return obs

    def convert_action(self, action: str) -> dict[str, str]:
        action_with_enum, *args = action.split("___")

        action, *enum_value_index = action_with_enum.split("^^^")

        value = 1 if len(enum_value_index) == 0 else int(enum_value_index[0])

        args = [a.replace("@", "") for a in args]

        key = f"{action}___{args[0]}" if len(args) > 0 else action

        return (key, value)

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
        actions = dict([self.convert_action(a) for a in actions])

        obs, reward, terminated, truncated, info = self.env.step(actions)

        obs = self.transform_obs(obs)

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed)

        obs = self.transform_obs(obs)

        return obs, info
