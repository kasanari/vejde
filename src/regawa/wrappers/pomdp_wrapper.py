import logging
from functools import cache
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gymnasium.spaces import Discrete, Dict, MultiDiscrete, Box

from .gym_utils import action_space, obs_space

from .utils import (
    HeteroGraph,
    create_render_graph,
    create_stacked_graphs,
    to_dict_action,
    create_stacked_obs,
)

from .utils import to_graphviz_alt, to_graphviz

from regawa import BaseModel

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
        self.wrapped_model = model
        self.env = env
        self.last_obs: dict[str, Any] = {}
        self.iter = 0
        self._idx_to_object: list[str] = ["None"]
        self._idx_to_object_type: list[str] = ["None"]

    def idx_to_object(self, idx: int) -> str:
        try:
            return self._idx_to_object[idx]
        except IndexError:
            return "None"

    def idx_to_object_type(self, idx: int) -> str:
        try:
            return self._idx_to_object_type[idx]
        except IndexError:
            return "None"

    @property
    def action_space(self) -> gym.spaces.MultiDiscrete:  # type: ignore
        return action_space(
            self.wrapped_model.action_fluents,
            self.wrapped_model.num_actions,
            len(self._idx_to_object) or 1,
            self.wrapped_model.arity,
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
                self.wrapped_model.idx_to_type,  # type: ignore
                self.wrapped_model.idx_to_fluent,  # type: ignore
            )

    @property
    @cache
    def observation_space(self) -> spaces.Dict:  # type: ignore
        # num_groundings = len(self.wrapped_model.groundings)
        # num_objects = self.wrapped_model.num_objects
        num_types = self.wrapped_model.num_types
        num_relations = self.wrapped_model.num_fluents
        max_arity = max(self.wrapped_model.arity(r) for r in self.wrapped_model.fluents)

        bool_space = Discrete(2)
        number_space = Box(
            np.finfo(np.float32).min,
            np.finfo(np.float32).max,
            shape=(),
        )

        bool_space = bool_space
        number_space = number_space

        return spaces.Dict(
            {
                "bool": obs_space(num_relations, num_types, max_arity, bool_space),
                "float": obs_space(num_relations, num_types, max_arity, number_space),
            }
        )

    def _create_obs(
        self, rddl_obs: dict[str, list[Any]]
    ) -> tuple[dict[str, Any], HeteroGraph]:
        g, _ = create_stacked_graphs(
            rddl_obs,
            self.wrapped_model,
        )
        o = create_stacked_obs(
            g,
            self.wrapped_model,
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

        info["state"] = g
        info["rddl_state"] = self.env.unwrapped.state  # type: ignore
        info["rddl_obs"] = rddl_obs

        self.last_obs = obs
        self.last_g = combined_g
        self.last_rddl_obs = rddl_obs
        self._idx_to_object_type = g.boolean.factor_values
        self._idx_to_object = g.boolean.factors

        return obs, info

    def _to_rddl_action(
        self, action: spaces.MultiDiscrete
    ) -> tuple[dict[str, int], str]:
        return to_dict_action(
            action,
            self.wrapped_model.idx_to_action,
            self.idx_to_object_type,
            self.idx_to_object,
            self.wrapped_model.fluent_params,
        )

    def step(
        self, action: spaces.MultiDiscrete
    ) -> tuple[spaces.Dict, SupportsFloat, bool, bool, dict[str, Any]]:
        rddl_action, grounded_action = self._to_rddl_action(
            action,
        )
        rddl_obs, reward, terminated, truncated, info = self.env.step(rddl_action)

        obs, g = self._create_obs(rddl_obs)

        info["state"] = g
        info["rddl_state"] = self.env.unwrapped.state  # type: ignore
        info["rddl_obs"] = rddl_obs
        info["rddl_action"] = rddl_action

        self.last_obs = obs
        self.last_g = create_render_graph(g.boolean, g.numeric)
        self.last_rddl_obs = rddl_obs
        self.last_action = grounded_action
        self._idx_to_object_type = g.boolean.factor_values  # type: ignore
        self._idx_to_object = g.boolean.factors  # type: ignore

        return obs, reward, terminated, truncated, info
