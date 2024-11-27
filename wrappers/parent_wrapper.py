import random
from copy import copy
from functools import cache
from typing import Any

from gymnasium.spaces import MultiDiscrete, Dict
import gymnasium as gym
import pyRDDLGym  # type: ignore
from pyRDDLGym.core.compiler.model import RDDLLiftedModel  # type: ignore

from wrappers.stacking_wrapper import StackingWrapper

from .utils import predicate, to_graphviz_alt


def get_groundings(model: RDDLLiftedModel, fluents: dict[str, Any]) -> set[str]:
    return set(
        dict(model.ground_vars_with_values(fluents)).keys()  # type: ignore
    )


class RDDLGraphWrapper(gym.Env[Dict, MultiDiscrete]):
    metadata = {"render_modes": ["human", "idx"]}

    def __init__(
        self,
        domain: str,
        instance: int,
        render_mode: str = "human",
        pomdp: bool = False,
    ) -> None:
        env: gym.Env = pyRDDLGym.make(domain, instance, enforce_action_constraints=True)  # type: ignore
        model: RDDLLiftedModel = env.model  # type: ignore
        object_to_type: dict[str, str] = copy(model.object_to_type)  # type: ignore
        types = set(object_to_type.values())  # type: ignore
        type_list = sorted(types)

        object_terms: list[str] = list(model.object_to_index.keys())  # type: ignore
        object_list = sorted(object_terms)

        self.pomdp = pomdp

        self.instance = instance
        self.domain = domain
        self.model = model
        self.num_types = len(type_list)
        self.idx_to_obj = object_list
        self.idx_to_type = type_list
        self.obj_to_type: dict[str, str] = object_to_type
        self.env: gym.Env[Dict, MultiDiscrete] = StackingWrapper(env) if pomdp else env

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
    def num_actions(self) -> int:
        return self.action_space.nvec[0]

    @property
    @cache
    def non_fluent_values(self) -> dict[str, int]:
        model = self.model
        return dict(
            model.ground_vars_with_values(model.non_fluents)  # type: ignore
        )

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
    def relations(self) -> list[str]:
        relation_list = sorted(set(predicate(g) for g in self.groundings))
        return relation_list

    @property
    @cache
    def groundings(self) -> list[str]:
        model = self.model

        state_fluents = model.state_fluents  # type: ignore
        action_fluents = model.action_fluents  # type: ignore
        observ_fluents = model.observ_fluents  # type: ignore
        interm_fluents = model.interm_fluents  # type: ignore

        action_groundings: set[str] = get_groundings(model, action_fluents)  # type: ignore
        observ_groundings = get_groundings(model, observ_fluents)  # type: ignore
        state_groundings: set[str] = get_groundings(model, state_fluents)  # type: ignore
        interm_groundings: set[str] = get_groundings(model, interm_fluents)  # type: ignore

        g: set[str] = set(
            g
            for _, v in model.variable_groundings.items()  # type: ignore
            for g in v  # type: ignore
            if g[-1] != model.NEXT_STATE_SYM  # type: ignore
        )

        g -= action_groundings
        g -= interm_groundings
        if self.pomdp:
            g -= state_groundings
            g |= observ_groundings

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
        return len(self.relations)

    @property
    @cache
    def num_objects(self) -> int:
        return len(self.idx_to_obj)

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
            symb: idx + 1 for idx, symb in enumerate(self.relations)
        }  # 0 is reserved for padding

    @property
    @cache
    def obj_to_idx(self) -> dict[str, int]:
        return {
            symb: idx + 1 for idx, symb in enumerate(self.idx_to_obj)
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
                    edge_indices,
                    edge_attributes,
                    self.idx_to_type,
                    self.idx_to_rel,
                )
            )
