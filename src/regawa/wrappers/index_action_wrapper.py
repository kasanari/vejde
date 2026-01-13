import logging
from typing import Any, SupportsFloat

import gymnasium as gym

from regawa.model import GroundObs, Grounding
from regawa.model import BaseModel
from regawa.model.null import NullConst
from ..model.grounding_func import to_dict_action
from regawa.data.heterograph import HeteroGraph
from collections.abc import Callable
import numpy as np
from ..data.actions import idx_action_to_ground_value
from gymnasium.spaces import MultiDiscrete

logger = logging.getLogger(__name__)


def action_space(
    action_fluents: tuple[str, ...],
    num_actions: int,
    num_objects: int,
    arity: Callable[[str], int],
) -> MultiDiscrete:
    max_action_args = max(arity(a) for a in action_fluents) or 1

    return MultiDiscrete(
        np.asarray(
            [num_actions]
            + [
                num_objects,
            ]
            * max_action_args
        )
    )


class IndexActionWrapper(
    gym.Wrapper[HeteroGraph, tuple[int, ...], HeteroGraph, GroundObs]
):
    """
    Converts actions from index-based to string-based
    """

    def __init__(self, env: gym.Env[HeteroGraph, GroundObs], model: BaseModel) -> None:
        super().__init__(env)
        self.env = env
        self.model = model
        self._idx_to_object = [NullConst.id]
        self._object_to_type: dict[str, str] = {NullConst.id: NullConst.type}

    def idx_to_object(self, idx: int) -> str:
        try:
            return self._idx_to_object[idx]
        except IndexError:
            logger.warning(f"Index {idx} not found in idx_to_object")
            return NullConst.id

    def _to_rddl_action(self, action: Grounding) -> GroundObs:
        return to_dict_action(action, self.obj_to_type, self.model.fluent_params)

    @property
    def action_space(self) -> gym.Space[MultiDiscrete]:  # type: ignore
        return action_space(  # type: ignore
            self.model.action_fluents,
            self.model.num_actions,
            len(self._object_to_type),
            self.model.arity,
        )

    @action_space.setter
    def action_space(self, space: gym.Space[MultiDiscrete]) -> None:  # type: ignore
        raise AttributeError("Can't set attribute")

    def step(
        self,
        action: tuple[int, ...],
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
        info["action_fluents"] = self.model.action_fluents

        self._idx_to_object = graph.boolean.factors.names
        self._object_to_type = dict(
            zip(graph.boolean.factors.names, graph.boolean.factors.types)
        )

        return graph, r, term, trunc, info

    def obj_to_type(self, obj: str) -> str:
        obj_type = self._object_to_type.get(obj, None)
        if obj_type is None:
            logger.warning(f"Object '{obj}' not found in object-to-type mapping.")
            return NullConst.type
        return obj_type

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[HeteroGraph, dict[str, Any]]:
        graph, info = self.env.reset(seed=seed)

        info["action_fluents"] = self.model.action_fluents

        self._idx_to_object = graph.boolean.factors.names
        self._object_to_type = dict(
            zip(graph.boolean.factors.names, graph.boolean.factors.types)
        )

        return graph, info
