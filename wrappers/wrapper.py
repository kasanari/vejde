import logging
import random
from functools import cache
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .parent_wrapper import RDDLGraphWrapper
from .utils import (
    generate_bipartite_obs,
    map_graph_to_idx,
    predicate,
    to_graphviz,
    FactorGraph,
)

logger = logging.getLogger(__name__)


def skip_fluent(key: str, variable_ranges: dict[str, str]) -> bool:
    return variable_ranges[predicate(key)] != "bool" or key == "noop"


class GroundedRDDLGraphWrapper(RDDLGraphWrapper):
    @property
    @cache
    def groundings(self):
        return sorted(
            [g for g in super().groundings if not skip_fluent(g, self.variable_ranges)]
        )

    @property
    @cache
    def observation_space(self) -> spaces.Dict:
        num_groundings = len(self.groundings)
        num_objects = self.num_objects
        num_types = self.num_types
        num_relations = len(self.rel_to_idx)
        num_edges = self.num_edges
        return spaces.Dict(
            {
                "var_type": spaces.Box(
                    low=0,
                    high=num_relations,
                    shape=(num_groundings,),
                    dtype=np.int64,
                ),
                "var_value": spaces.Box(
                    low=0,
                    high=1,
                    shape=(num_groundings,),
                    dtype=np.bool_,
                ),
                "factor": spaces.Box(
                    low=0, high=num_types, shape=(num_objects,), dtype=np.int64
                ),
                "edge_index": spaces.Box(
                    low=0,
                    high=max(num_objects, num_groundings),
                    shape=(num_edges, 2),
                    dtype=np.int64,
                ),
                "edge_attr": spaces.Box(
                    low=0,
                    high=1,
                    shape=(num_edges,),
                    dtype=np.int64,
                ),
                "length": spaces.Box(
                    low=0,
                    high=40,
                    shape=(num_groundings,),
                    dtype=np.int64,
                ),
            }
        )

    def _create_obs(
        self, rddl_obs: dict[str, Any]
    ) -> tuple[dict[str, Any], FactorGraph]:
        non_fluent_values = {
            k: [v] + [None] * 40 for k, v in self.non_fluent_values.items()
        }

        rddl_obs |= non_fluent_values

        filtered_groundings = sorted(
            [g for g in self.groundings if not skip_fluent(g, self.variable_ranges)]
        )

        filtered_obs: dict[str, Any] = {k: rddl_obs[k] for k in filtered_groundings}

        g = generate_bipartite_obs(
            filtered_obs,
            filtered_groundings,
            self.obj_to_type,
            self.variable_ranges,
        )

        idx_g = map_graph_to_idx(g, self.rel_to_idx, self.type_to_idx)

        obs = {
            "var_type": idx_g.variables,
            "var_value": idx_g.values,
            "factor": idx_g.factors,
            "edge_index": idx_g.edge_indices,
            "edge_attr": idx_g.edge_attributes,
            # "numeric": numeric,
        }
        return obs, g

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[spaces.Dict, dict[str, Any]]:
        rddl_obs, info = self.env.reset(seed=seed)

        # obs |= self.action_values

        rddl_obs, lengths = rddl_obs

        obs, g = self._create_obs(rddl_obs)

        self.iter = 0
        self.last_obs = obs
        self.last_rddl_obs = rddl_obs

        info["state"] = g

        non_fluent_lengths = {k: 1 for k in self.non_fluent_values}
        lengths |= non_fluent_lengths
        length = np.array([lengths[key] for key in self.groundings], dtype=np.int64)
        obs["length"] = length

        return obs, info

    def step(self, action: int | list[int]):
        action_fluent = self.action_fluents[action[0]]
        object_id = self.idx_to_obj[action[1]]

        rddl_action = (
            f"{action_fluent}___{object_id}" if action_fluent != "noop" else "noop"
        )

        self.last_action = rddl_action

        invalid_action = rddl_action not in self.action_groundings

        if invalid_action:
            logger.warning(f"Invalid action: {rddl_action}")

        rddl_action_dict = (
            {} if invalid_action or action_fluent == "noop" else {rddl_action: 1}
        )

        rddl_obs, reward, terminated, truncated, info = self.env.step(rddl_action_dict)
        rddl_obs, lengths = rddl_obs
        obs, g = self._create_obs(rddl_obs)

        self.iter += 1
        self.last_obs = obs
        self.last_rddl_obs = rddl_obs
        self.last_action_values = rddl_action_dict

        info["state"] = g

        non_fluent_lengths = {k: 1 for k in self.non_fluent_values}
        lengths |= non_fluent_lengths
        length = np.array([lengths[key] for key in self.groundings], dtype=np.int64)
        obs["length"] = length

        return obs, reward, terminated, truncated, info


def register_env():
    env_id = "GroundedRDDLGraphWrapper-v0"
    gym.register(
        id=env_id,
        entry_point="wrappers.wrapper:GroundedRDDLGraphWrapper",
    )
    return env_id
