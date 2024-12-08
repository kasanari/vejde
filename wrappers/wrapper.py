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
        model: RDDLLiftedModel = env.model  # type: ignore

        self.instance = instance
        self.domain = domain
        self.model = model
        self.env: RDDLEnv = env
        self.last_obs: dict[str, Any] = {}
        self.iter = 0

    @property
    @cache
    def action_space(self) -> gym.spaces.MultiDiscrete:  # type: ignore
        action_fluents = self.model.action_fluents  # type: ignore
        return gym.spaces.MultiDiscrete(
            [
                len(action_fluents) + 1,  # type: ignore
                self.num_objects,
            ]
        )

    @property
    @cache
    def idx_to_type(self) -> list[str]:
        return sorted(set(self.obj_to_type.values()))

    @property
    @cache
    def obj_to_type(self) -> dict[str, str]:
        model: RDDLLiftedModel = self.model  # type: ignore
        object_to_type: dict[str, str] = copy(model.object_to_type)  # type: ignore
        return object_to_type

    @property
    @cache
    def num_types(self) -> int:
        return len(self.idx_to_type)

    @property
    @cache
    def num_actions(self) -> int:
        return self.action_space.nvec[0]

    @property
    @cache
    def non_fluent_values(self) -> dict[str, int]:
        # model = self.model
        # return dict(
        #     model.ground_vars_with_values(model.non_fluents)  # type: ignore
        # )

        nf_vals = {}

        non_fluents = self.model.ast.non_fluents.init_non_fluent  # type: ignore
        for (name, params), value in non_fluents:
            gname = RDDLPlanningModel.ground_var(name, params)
            nf_vals[gname] = value

        return nf_vals

    @property
    @cache
    def num_edges(self) -> int:
        return sum(self.arities[predicate(g)] for g in self.groundings)

    @property
    @cache
    def variable_ranges(self) -> dict[str, str]:
        variable_ranges: dict[str, str] = self.model._variable_ranges  # type: ignore
        variable_ranges["noop"] = "bool"
        return variable_ranges

    @property
    @cache
    def variable_params(self) -> dict[str, list[str]]:
        variable_params: dict[str, list[str]] = copy(self.model.variable_params)  # type: ignore
        variable_params["noop"] = []
        return variable_params

    @property
    @cache
    def type_to_arity(self) -> dict[str, int]:
        vp = self.model.variable_params  # type: ignore
        return {
            value[0]: [k for k, v in vp.items() if v == value]  # type: ignore
            for _, value in vp.items()  # type: ignore
            if len(value) == 1  # type: ignore
        }

    @property
    @cache
    def arities_to_fluent(self) -> dict[int, list[str]]:
        arities: dict[str, int] = self.arities
        return {
            value: [k for k, v in arities.items() if v == value]
            for _, value in arities.items()
        }

    @property
    @cache
    def idx_to_object(self) -> list[str]:
        object_terms: list[str] = list(self.model.object_to_index.keys())  # type: ignore
        object_list = sorted(object_terms)
        return object_list

    @property
    @cache
    def idx_to_relation(self) -> list[str]:
        relation_list = sorted(set(predicate(g) for g in self.groundings))
        return relation_list

    @property
    @cache
    def all_groundings(self) -> list[str]:
        model = self.model

        state_fluents = model.state_fluents  # type: ignore

        non_fluent_groundings = set(self.non_fluent_values.keys())
        state_groundings: set[str] = get_groundings(model, state_fluents)  # type: ignore

        g = state_groundings | non_fluent_groundings

        return sorted(g)

    @property
    @cache
    def action_fluents(self) -> list[str]:
        model = self.model
        action_fluents = model.action_fluents  # type: ignore
        return ["noop"] + sorted(action_fluents)  # type: ignore

    @property
    @cache
    def action_groundings(self) -> set[str]:
        return get_groundings(self.model, self.model.action_fluents) | {"noop"}  # type: ignore

    @property
    @cache
    def num_relations(self) -> int:
        return len(self.idx_to_relation)

    @property
    @cache
    def num_objects(self) -> int:
        return len(self.idx_to_object)

    @property
    @cache
    def type_to_idx(self) -> dict[str, int]:
        return {
            symb: idx + 1 for idx, symb in enumerate(self.idx_to_type)
        }  # 0 is reserved for padding

    @property
    @cache
    def rel_to_idx(self) -> dict[str, int]:
        return {
            symb: idx + 1 for idx, symb in enumerate(self.idx_to_relation)
        }  # 0 is reserved for padding

    @property
    @cache
    def obj_to_idx(self) -> dict[str, int]:
        return {
            symb: idx + 1 for idx, symb in enumerate(self.idx_to_object)
        }  # 0 is reserved for padding

    @property
    @cache
    def arities(self) -> dict[str, int]:
        return {key: len(value) for key, value in self.variable_params.items()}

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
                    self.idx_to_type,
                    self.idx_to_relation,
                )
            )

    @property
    @cache
    def groundings(self):
        return sorted(
            [g for g in self.all_groundings if not skip_fluent(g, self.variable_ranges)]
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
            self.non_fluent_values,
            self.rel_to_idx,
            self.type_to_idx,
            self.groundings,
            self.obj_to_type,
            self.variable_ranges,
            skip_fluent,
        )

        o["length"] = np.ones(len(self.groundings))
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
