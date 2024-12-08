import logging
from functools import cache
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .utils import predicate, to_rddl_action, create_obs
import pyRDDLGym  # type: ignore
from pyRDDLGym.core.compiler.model import RDDLLiftedModel, RDDLPlanningModel  # type: ignore
from pyRDDLGym import RDDLEnv  # type: ignore
from .utils import to_graphviz_alt, get_groundings
from copy import copy

from .utils import to_graphviz_alt
from .rddl_model import RDDLModel

logger = logging.getLogger(__name__)


def skip_fluent(key: str, variable_ranges: dict[str, str]) -> bool:
    return variable_ranges[predicate(key)] != "bool" or key == "noop"


class GroundedRDDLGraphWrapper(gym.Env):
    metadata = {"render_modes": ["human", "idx"]}

    def __init__(
        self,
        domain: str,
        instance: int,
        render_mode: str = "human",
    ) -> None:
        env: gym.Env = pyRDDLGym.make(domain, instance, enforce_action_constraints=True)  # type: ignore
        wrapped_model = RDDLModel(env.model)  # type: ignore
        self.wrapped_model = wrapped_model

        self.instance = instance
        self.domain = domain
        self.env: RDDLEnv = env
        self.last_obs: dict[str, Any] = {}
        self.iter = 0

    @property
    @cache
    def action_space(self) -> gym.spaces.MultiDiscrete:  # type: ignore
        return gym.spaces.MultiDiscrete(
            [
                self.wrapped_model.num_actions,  # type: ignore
                self.wrapped_model.num_objects,
            ]
        )

    def render(self):
        obs = self.last_obs
        nodes_classes = obs["predicate_class"]
        node_values = obs["predicate_value"]
        object_nodes = obs["object"]
        edge_indices = obs["edge_index"]
        edge_attributes = obs["edge_attr"]
        # numeric = obs["numeric"]

        with open(f"{self.domain}_{self.instance}_{self.iter}.dot", "w") as f:
            f.write(
                to_graphviz_alt(
                    nodes_classes,
                    node_values,
                    object_nodes,
                    edge_indices,  # type: ignore
                    edge_attributes,
                    self.wrapped_model.idx_to_type,
                    self.wrapped_model.idx_to_relation,
                )
            )

    @property
    @cache
    def observation_space(self) -> spaces.Dict:  # type: ignore
        num_groundings = len(self.wrapped_model.groundings)
        num_objects = self.wrapped_model.num_objects
        num_types = self.wrapped_model.num_types
        num_relations = len(self.wrapped_model.rel_to_idx)
        num_edges = self.wrapped_model.num_edges

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
                shape=(2, num_edges),
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
                high=1,
                shape=(num_groundings,),
                dtype=np.int64,
            ),
        }

        return spaces.Dict(s)

    def create_obs(
        self, rddl_obs: dict[str, list[int]]
    ) -> tuple[spaces.Dict, dict[str, Any]]:
        o, g = create_obs(
            rddl_obs,
            self.wrapped_model.non_fluent_values,
            self.wrapped_model.rel_to_idx,
            self.wrapped_model.type_to_idx,
            self.wrapped_model.groundings,
            self.wrapped_model.obj_to_type,
            self.wrapped_model.variable_ranges,
            skip_fluent,
        )

        o["length"] = np.ones(len(self.wrapped_model.groundings))
        return o, g

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
            action,
            self.wrapped_model.action_fluents,
            self.wrapped_model.idx_to_object,
            self.wrapped_model.action_groundings,
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
