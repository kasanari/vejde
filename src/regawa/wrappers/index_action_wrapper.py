import logging
from typing import Any, SupportsFloat

import gymnasium as gym

from regawa.model.base_grounded_model import GroundObs, Grounding
from regawa.model.base_model import BaseModel
from regawa.wrappers.grounding_utils import to_dict_action
from regawa.wrappers.types import HeteroGraph
from .gym_utils import action_space
from .utils import idx_action_to_ground_value
from gymnasium.spaces import MultiDiscrete
logger = logging.getLogger(__name__)


class IndexActionWrapper(
    gym.Wrapper[HeteroGraph, MultiDiscrete, HeteroGraph, GroundObs]
):
    """
    Converts actions from index-based to string-based
    """

    def __init__(
        self, env: gym.Env[HeteroGraph, GroundObs], model: BaseModel
    ) -> None:
        super().__init__(env)
        self.env = env
        self.model = model
        self._idx_to_object = ["None"]
        self._object_to_type: dict[str, str] = {"None": "None"}

    def idx_to_object(self, idx: int) -> str:
        try:
            return self._idx_to_object[idx]
        except IndexError:
            logger.warning(f"Index {idx} not found in idx_to_object")
            return "None"

    def _to_rddl_action(self, action: Grounding) -> GroundObs:
        return to_dict_action(action, self.obj_to_type, self.model.fluent_params)

    @property
    def action_space(self) -> gym.Space[MultiDiscrete]:
        return action_space(
            self.model.action_fluents,
            self.model.num_actions,
            len(self._object_to_type),
            self.model.arity,
        )
    
    @action_space.setter
    def action_space(self, space: gym.Space[MultiDiscrete]) -> None:
        raise AttributeError("Can't set attribute")

    def step(
        self,
        action: MultiDiscrete,
    ) -> tuple[
        HeteroGraph,
        SupportsFloat,
        bool,
        bool,
        dict[str, Any],
    ]:
        a = idx_action_to_ground_value(
            action, self.model.idx_to_action, self.idx_to_object
        )

        rddl_action = self._to_rddl_action(a)

        graph, r, term, trunc, info = self.env.step(rddl_action)

        info["rddl_action"] = rddl_action

        self._idx_to_object = graph.boolean.factors
        self._object_to_type = dict(zip(graph.boolean.factors, graph.boolean.factor_types))

        return graph, r, term, trunc, info
    
    def obj_to_type(self, obj: str) -> str:
        obj_type = self._object_to_type.get(obj, None)
        if obj_type is None:
            logger.warning(f"Object '{obj}' not found in object-to-type mapping.")
            return "None"
        return obj_type

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[HeteroGraph, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        graph, info = self.env.reset(seed=seed)

        self._idx_to_object = graph.boolean.factors
        self._object_to_type = dict(zip(graph.boolean.factors, graph.boolean.factor_types))

        return graph, info
