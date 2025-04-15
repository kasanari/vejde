import logging
from functools import cached_property
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete

from regawa import BaseModel
from regawa.model import GroundValue

from .graph_utils import create_graphs_func, create_obs_dict_func
from .grounding_utils import to_dict_action
from .gym_utils import action_space, obs_space
from .render_utils import create_render_graph, to_graphviz, to_graphviz_alt
from .util_types import HeteroGraph, RenderGraph

logger = logging.getLogger(__name__)


class GroundedGraphWrapper(
    gym.Wrapper[dict[GroundValue, Any], MultiDiscrete, Dict, Dict]
):
    metadata: dict[str, Any] = {"render_modes": ["human", "idx"]}

    def __init__(
        self,
        env: gym.Env[dict[str, Any], dict[str, int]],
        model: BaseModel,
        render_mode: str = "human",
    ) -> None:
        super().__init__(env)
        self.model = model
        self.last_obs: dict[str, Any] = {}
        self.last_action: GroundValue | None = None
        self.last_g: RenderGraph | None = None
        self._object_to_type: dict[str, str] = {"None": "None"}
        self.create_graphs = create_graphs_func(model)
        self.create_obs_dict = create_obs_dict_func(model)

    @property
    def action_space(self) -> MultiDiscrete:
        return action_space(
            self.model.action_fluents,
            self.model.num_actions,
            len(self._object_to_type),
            self.model.arity,
        )

    def render(self):
        if self.metadata.get("render_modes") == "idx":
            obs = self.last_obs
            return to_graphviz_alt(
                obs["var_type"],
                obs["var_value"],
                obs["factor"],
                obs["edge_index"].T,
                obs["edge_attr"],
                self.model.idx_to_type,
                self.model.idx_to_fluent,
            )

        # Default rendering mode
        return to_graphviz(self.last_g, scaling=10)

    @cached_property
    def observation_space(self) -> Dict:
        num_types = self.model.num_types
        num_relations = self.model.num_fluents
        max_arity = max(self.model.arity(r) for r in self.model.fluents)
        num_actions = self.model.num_actions

        bool_space = Discrete(2)
        number_space = Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(),
        )

        return Dict(
            {
                "bool": obs_space(
                    num_relations, num_types, max_arity, num_actions, bool_space
                ),
                "float": obs_space(
                    num_relations, num_types, max_arity, num_actions, number_space
                ),
            }
        )

    def _create_obs(
        self, rddl_observation: dict[GroundValue, Any]
    ) -> tuple[dict[str, Any], HeteroGraph]:
        graph, _ = self.create_graphs(rddl_observation)
        obs_dict = self.create_obs_dict(graph)
        return obs_dict, graph

    def _prepare_info(
        self,
        rddl_obs: dict[GroundValue, Any],
        graph: HeteroGraph,
        rddl_action: dict[GroundValue, Any],
    ) -> tuple[dict[str, Any], RenderGraph, dict[str, str]]:
        combined_graph = create_render_graph(graph.boolean, graph.numeric)
        object_to_type = dict(zip(graph.boolean.factors, graph.boolean.factor_types))
        info: dict[str, Any] = {
            "state": combined_graph,
            "rddl_state": rddl_obs,
            "idx_to_object": graph.boolean.factors,
            "object_to_type": object_to_type,
            "action_fluents": self.model.action_fluents,
            "rddl_action": rddl_action,
        }
        return info, combined_graph, object_to_type

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Dict, dict[str, Any]]:
        super().reset(seed=seed)
        rddl_obs, info = self.env.reset(seed=seed)

        obs, graph = self._create_obs(rddl_obs)
        info_update, combined_graph, object_to_type = self._prepare_info(
            rddl_obs, graph, {}
        )
        info.update(info_update)

        self.last_obs = obs
        self.last_g = combined_graph
        self._object_to_type = object_to_type
        self.last_action = None

        return obs, info

    def obj_to_type(self, obj: str) -> str:
        obj_type = self._object_to_type.get(obj, None)
        if obj_type is None:
            logger.warning(f"Object '{obj}' not found in object-to-type mapping.")
            return "None"
        return obj_type

    def _to_rddl_action(self, action: GroundValue) -> dict[GroundValue, Any]:
        return to_dict_action(action, self.obj_to_type, self.model.fluent_params)

    def step(
        self, action: GroundValue
    ) -> tuple[Dict, SupportsFloat, bool, bool, dict[str, Any]]:
        rddl_action = self._to_rddl_action(action)
        rddl_obs, reward, terminated, truncated, info = self.env.step(rddl_action)

        obs, graph = self._create_obs(rddl_obs)
        info_update, combined_graph, object_to_type = self._prepare_info(
            rddl_obs, graph, rddl_action
        )
        info.update(info_update)
        self._object_to_type = object_to_type
        self.last_obs = obs
        self.last_g = combined_graph
        self.last_rddl_obs = rddl_obs
        self.last_action = action

        return obs, reward, terminated, truncated, info
