import logging
from functools import cache
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete

from regawa import BaseModel
from regawa.model import GroundValue

from .grounding_utils import to_dict_action
from .gym_utils import action_space, obs_space
from .render_utils import create_render_graph, to_graphviz, to_graphviz_alt
from .stacked_utils import create_graphs, create_obs_dict
from .util_types import HeteroGraph

logger = logging.getLogger(__name__)


class StackingGroundedGraphWrapper(
    gym.Wrapper[dict[str, Any], MultiDiscrete, Dict, Dict]
):
    @property
    def metadata(self) -> dict[str, Any]:
        return {"render_modes": ["human", "idx"]}

    @metadata.setter
    def metadata(self, value: dict[str, Any]):
        self._metadata = value

    def __init__(
        self,
        env: gym.Env[dict[str, list[Any]], dict[str, int]],
        model: BaseModel,
        render_mode: str = "human",
    ) -> None:
        super().__init__(env)
        self.model = model
        self.env = env
        self.last_obs: dict[str, Any] = {}
        self.last_action: GroundValue | None = None
        self.iter = 0
        self._object_to_type: dict[str, str] = {"None": "None"}

    @property
    def action_space(self) -> gym.spaces.MultiDiscrete:  # type: ignore
        return action_space(
            self.model.action_fluents,
            self.model.num_actions,
            len(self._object_to_type) or 1,
            self.model.arity,
        )

    def render(self):
        return to_graphviz(self.last_g, scaling=10)

        if self.metadata["render_modes"] == "idx":
            obs = self.last_obs
            nodes_classes = obs["var_type"]
            node_values = obs["var_value"]
            object_nodes = obs["factor"]
            edge_indices = obs["edge_index"].T
            edge_attributes = obs["edge_attr"]
            # numeric = obs["numeric"]

            return to_graphviz_alt(
                nodes_classes,
                node_values,
                object_nodes,
                edge_indices,  # type: ignore
                edge_attributes,
                self.model.idx_to_type,  # type: ignore
                self.model.idx_to_fluent,  # type: ignore
            )

    @property
    @cache
    def observation_space(self) -> spaces.Dict:  # type: ignore
        # num_groundings = len(self.wrapped_model.groundings)
        # num_objects = self.wrapped_model.num_objects
        num_types = self.model.num_types
        num_relations = self.model.num_fluents
        max_arity = max(self.model.arity(r) for r in self.model.fluents)
        num_actions = self.model.num_actions

        bool_space = Discrete(2)
        number_space = Box(
            np.finfo(np.float32).min,
            np.finfo(np.float32).max,
            shape=(),
        )

        return spaces.Dict(
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
        self, rddl_obs: dict[str, list[Any]]
    ) -> tuple[dict[str, Any], HeteroGraph]:
        g, _ = create_graphs(
            rddl_obs,
            self.model,
        )
        o = create_obs_dict(
            g,
            self.model,
        )

        assert o["bool"]["length"].sum() == len(
            o["bool"]["var_value"]
        ), "Expected {} but got {}".format(
            o["bool"]["length"].sum(), len(o["bool"]["var_value"])
        )
        assert o["float"]["length"].sum() == len(
            o["float"]["var_value"]
        ), "Expected {} but got {}".format(
            o["float"]["length"].sum(), len(o["float"]["var_value"])
        )

        return o, g

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[spaces.Dict, dict[str, Any]]:
        super().reset(seed=seed)
        rddl_obs, info = self.env.reset(seed=seed)

        obs, g = self._create_obs(rddl_obs)

        combined_g = create_render_graph(g.boolean, g.numeric)

        info["state"] = combined_g
        info["rddl_state"] = (
            self.env.unwrapped.state if hasattr(self.env.unwrapped, "state") else {}
        )  # type: ignore
        info["rddl_obs"] = rddl_obs
        info["idx_to_object"] = g.boolean.factors
        info["action_fluents"] = self.model.action_fluents

        self._object_to_type = {
            k: v for k, v in zip(g.boolean.factors, g.boolean.factor_types)
        }

        self.last_obs = obs
        self.last_g = combined_g
        self.last_rddl_obs = rddl_obs
        self.last_action = None

        return obs, info

    def obj_to_type(self, obj: str) -> str:
        try:
            return self._object_to_type[obj]
        except IndexError:
            logger.warning(f"Object {obj} not found in object_to_type")
            return "None"

    def _to_rddl_action(self, action: GroundValue) -> dict[GroundValue, Any]:
        return to_dict_action(action, self.obj_to_type, self.model.fluent_params)

    def step(
        self, action: GroundValue
    ) -> tuple[spaces.Dict, SupportsFloat, bool, bool, dict[str, Any]]:
        rddl_action = self._to_rddl_action(action)
        rddl_obs, reward, terminated, truncated, info = self.env.step(rddl_action)

        obs, g = self._create_obs(rddl_obs)

        combined_g = create_render_graph(g.boolean, g.numeric)

        info["state"] = combined_g
        info["rddl_state"] = (
            self.env.unwrapped.state if hasattr(self.env.unwrapped, "state") else {}
        )  # type: ignore
        info["rddl_obs"] = rddl_obs
        info["rddl_action"] = rddl_action
        info["idx_to_object"] = g.boolean.factors
        info["action_fluents"] = self.model.action_fluents

        self._object_to_type = {
            k: v for k, v in zip(g.boolean.factors, g.boolean.factor_types)
        }

        self.last_obs = obs
        self.last_g = combined_g
        self.last_rddl_obs = rddl_obs
        self.last_action = action

        return obs, reward, terminated, truncated, info
