from typing import Any, TypeVar
import gymnasium as gym
from gymnasium import spaces
from pyRDDLGym import RDDLEnv

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
WrapperObsType = spaces.Dict
WrapperActType = spaces.Tuple


class RDDLDefaultInvalidActions(
    gym.Wrapper[WrapperActType, WrapperObsType, ObsType, ActType]
):
    def __init__(self, env: RDDLEnv) -> None:
        super().__init__(env)
        self.env = env

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
        is_valid = self.env.sampler.check_action_preconditions(actions, silent=True)

        actions = actions if is_valid else {}

        return self.env.step(actions)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        super().reset(seed=seed)
        return self.env.reset(seed=seed)
