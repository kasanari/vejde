import logging
from functools import cache
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .parent_wrapper import RDDLGraphWrapper
from .utils import predicate, to_rddl_action, create_obs

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
    def observation_space(self) -> spaces.Dict:  # type: ignore
        num_groundings = len(self.groundings)
        num_objects = self.num_objects
        num_types = self.num_types
        num_relations = len(self.rel_to_idx)
        num_edges = self.num_edges

        s: dict[str, spaces.Space] = {  # type: ignore
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
                dtype=np.int8,
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
        }

        return spaces.Dict(s)

    def create_obs(
        self, rddl_obs: dict[str, list[int]]
    ) -> tuple[spaces.Dict, dict[str, Any]]:
        return create_obs(
            rddl_obs,
            self.non_fluent_values,
            self.rel_to_idx,
            self.type_to_idx,
            self.groundings,
            self.obj_to_type,
            self.variable_ranges,
            skip_fluent,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[spaces.Dict, dict[str, Any]]:
        rddl_obs, info = self.env.reset(seed=seed)

        obs, g = self.create_obs(rddl_obs)

        info["state"] = g
        info["rddl_state"] = self.env.state  # type: ignore

        self.last_obs = obs
        self.last_rddl_obs = rddl_obs

        return obs, info

    def step(
        self, action: spaces.MultiDiscrete
    ) -> tuple[spaces.Dict, SupportsFloat, bool, bool, dict[str, Any]]:
        rddl_action_dict, rddl_action = to_rddl_action(
            action, self.action_fluents, self.idx_to_object, self.action_groundings
        )
        rddl_obs, reward, terminated, truncated, info = self.env.step(rddl_action_dict)

        obs, g = self.create_obs(rddl_obs)

        info["state"] = g
        info["rddl_state"] = self.env.state  # type: ignore

        self.last_obs = obs
        self.last_rddl_obs = rddl_obs
        self.last_action_values = rddl_action_dict
        self.last_action = rddl_action

        return obs, reward, terminated, truncated, info


def register_env():
    env_id = "GroundedRDDLGraphWrapper-v0"
    gym.register(
        id=env_id,
        entry_point="wrappers.wrapper:GroundedRDDLGraphWrapper",
    )
    return env_id
