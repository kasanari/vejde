import logging
from typing import Any, TypeVar

import gymnasium as gym

from regawa.model.base_model import BaseModel

from .utils import idx_action_to_ground_value

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")

logger = logging.getLogger(__name__)


class IndexActionWrapper(
    gym.Wrapper[gym.spaces.Tuple, gym.spaces.Dict, gym.spaces.Tuple, gym.spaces.Dict]
):
    """
    Adds actions to the previous observation.
    """

    def __init__(
        self, env: gym.Env[gym.spaces.Dict, gym.spaces.Dict], model: BaseModel
    ) -> None:
        super().__init__(env)
        self.env = env
        self.model = model
        self._idx_to_object = ["None"]

    def idx_to_object(self, idx: int) -> str:
        try:
            return self._idx_to_object[idx]
        except IndexError:
            logger.warning(f"Index {idx} not found in idx_to_object")
            return "None"

    def step(
        self,
        actions: tuple[int, ...],
    ) -> tuple[
        dict[str, bool | None],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        a = idx_action_to_ground_value(
            actions, self.model.idx_to_action, self.idx_to_object
        )

        *rest, info = self.env.step(a)

        self._idx_to_object = info["idx_to_object"]

        return *rest, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[gym.spaces.Dict, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        *rest, info = self.env.reset(seed=seed)

        self._idx_to_object = info["idx_to_object"]

        return *rest, info
