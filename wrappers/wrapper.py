import random
from .parent_wrapper import RDDLGraphWrapper
from .utils import generate_bipartite_obs, to_graphviz, predicate
import numpy as np
from typing import Any
import logging
from gymnasium import spaces


logger = logging.getLogger(__name__)


def skip_fluent(key: str, variable_ranges: dict[str, str]) -> bool:
    return variable_ranges[predicate(key)] != "bool" or key == "noop"


class GroundedRDDLGraphWrapper(RDDLGraphWrapper):
    def __init__(self, domain: str, instance: int, render_mode: str = "human") -> None:
        super().__init__(domain, instance, render_mode)

        filtered_groundings = [
            g for g in self.groundings if not skip_fluent(g, self.variable_ranges)
        ]
        num_groundings = len(filtered_groundings)
        num_objects = self.num_objects
        num_types = self.num_types
        num_relations = len(set(predicate(g) for g in filtered_groundings))
        num_edges = sum(self.arities[predicate(g)] for g in filtered_groundings)
        relation_list = sorted(set(predicate(g) for g in filtered_groundings))
        self.rel_to_idx = {
            k: i
            for i, k in enumerate(
                sorted(set(predicate(g) for g in filtered_groundings))
            )
        }
        self.idx_to_rel = relation_list
        self.observation_space = spaces.Dict(
            {
                "predicate_class": spaces.Box(
                    low=0,
                    high=num_relations,
                    shape=(num_groundings,),
                    dtype=np.int64,
                ),
                "predicate_value": spaces.Box(
                    low=0, high=1, shape=(num_groundings,), dtype=np.int64
                ),
                "object": spaces.Box(
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
            }
        )
        pass

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[spaces.Dict, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed)

        obs |= self.action_values
        obs |= self.non_fluents_values

        filtered_groundings = sorted(
            [g for g in self.groundings if not skip_fluent(g, self.variable_ranges)]
        )

        filtered_obs: dict[str, Any] = {k: obs[k] for k in filtered_groundings}

        (
            predicate_classes,
            predicate_values,
            object_nodes,
            edge_indices,
            edge_attributes,
            _,
        ) = generate_bipartite_obs(
            filtered_obs,
            filtered_groundings,
            self.rel_to_idx,
            self.type_to_idx,
            self.obj_to_type,
            self.variable_ranges,
        )

        obs = {
            "predicate_class": predicate_classes,
            "predicate_value": predicate_values,
            "object": object_nodes,
            # "numeric": numeric,
            "edge_index": edge_indices,
            "edge_attr": edge_attributes,
        }

        self.iter = 0
        self.last_obs = obs

        info["idx_to_obj"] = self.obj_to_idx
        info["idx_to_rel"] = self.rel_to_idx

        return obs, info

    def step(self, action: int | list[int]):
        grounding = self.groundings[action[0]]

        logger.debug(f"Action: {grounding}")

        rddl_action_dict = (
            {}
            if grounding not in self.action_groundings or grounding == "noop"
            else {grounding: 1}
        )

        obs, reward, terminated, truncated, info = self.env.step(rddl_action_dict)

        obs |= self.action_values
        obs |= self.non_fluents_values

        filtered_groundings = sorted(
            [g for g in self.groundings if not skip_fluent(g, self.variable_ranges)]
        )

        filtered_obs: dict[str, Any] = {k: obs[k] for k in filtered_groundings}

        (
            predicate_classes,
            predicate_values,
            object_nodes,
            edge_indices,
            edge_attributes,
            numeric,
        ) = generate_bipartite_obs(
            filtered_obs,
            filtered_groundings,
            self.rel_to_idx,
            self.type_to_idx,
            self.obj_to_type,
            self.variable_ranges,
        )

        obs = {
            "predicate_class": predicate_classes,
            "predicate_value": predicate_values,
            "object": object_nodes,
            "numeric": numeric,
            "edge_index": edge_indices,
            "edge_attr": edge_attributes,
        }

        self.iter += 1
        self.last_obs = obs

        info["idx_to_obj"] = self.obj_to_idx
        info["idx_to_rel"] = self.rel_to_idx

        return obs, reward, terminated, truncated, info
