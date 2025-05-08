import logging
import random
from collections.abc import Callable
from itertools import chain
from typing import Any, TypeVar, Generic

import numpy as np
from gymnasium.spaces import Dict
from numpy.typing import NDArray

from regawa.model import GroundValue
from regawa.wrappers.grounding_utils import (
    arity,
    create_edges,
    objects,
    objects_with_type_func,
    predicate,
)
from regawa.wrappers.util_types import (
    Edge,
    FactorGraph,
    IdxFactorGraph,
    Object,
    StackedFactorGraph,
    Variables,
)

logger = logging.getLogger(__name__)

V = TypeVar("V")
T = TypeVar(
    "T",
    FactorGraph[bool],
    StackedFactorGraph[bool],
    FactorGraph[float],
    StackedFactorGraph[float],
)


def map_graph_to_idx(
    variables: Variables[V],
    global_variables: Variables[V],
    senders: NDArray[np.int64],
    receivers: NDArray[np.int64],
    edge_attributes: list[int],
    action_type_mask: list[tuple[bool, ...]],
    action_arity_mask: list[tuple[bool, ...]],
    factor_types: list[str],
    rel_to_idx: Callable[[str], int],
    type_to_idx: Callable[[str], int],
    var_val_dtype: type,
) -> IdxFactorGraph[V]:
    arr = np.asarray
    factor_type_idx = arr(
        [type_to_idx(f_type) for f_type in factor_types], dtype=np.int64
    )
    idx_global_vars = arr(
        [rel_to_idx(p) for p in global_variables.types], dtype=np.int64
    )
    idx_vars = arr([rel_to_idx(p) for p in variables.types], dtype=np.int64)

    return IdxFactorGraph(
        Variables(
            idx_vars,
            arr(variables.values, dtype=var_val_dtype),
            arr(variables.lengths),
        ),
        factor_type_idx,
        senders,
        receivers,
        arr(edge_attributes, dtype=np.int64),
        Variables(
            idx_global_vars,
            arr(global_variables.values, dtype=var_val_dtype),
            arr(global_variables.lengths),
        ),
        arr(action_type_mask, dtype=np.bool_),
        arr(action_arity_mask, dtype=np.bool_),
    )


def from_dict_action(
    action: tuple[str, ...],
    action_to_idx: Callable[[str], int],
    obj_to_idx: Callable[[str], int],
) -> tuple[int, ...]:
    action_idx = action_to_idx(action[0])
    object_idxs = [obj_to_idx(obj) for obj in action[1:]]
    return (action_idx, *object_idxs)


def idx_action_to_ground_value(
    action: tuple[int, ...],
    idx_to_action: Callable[[int], str],
    idx_to_obj: Callable[[int], str],
) -> GroundValue:
    action_name = idx_to_action(action[0])
    o = tuple(idx_to_obj(obj_idx) for obj_idx in action[1:] if obj_idx != 0)
    return (action_name, *o)


def sample_action(action_space: Dict) -> dict[str, int]:
    action = action_space.sample()  # type: ignore
    chosen_action, value = random.choice(list(action.items()))  # type: ignore
    return {chosen_action: value}


def translate_edges(
    source_to_index: Callable[[GroundValue], int],
    target_to_index: Callable[[str], int],
    edges: list[Edge],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    senders = np.asarray([source_to_index(edge[0]) for edge in edges], dtype=np.int64)
    receivers = np.asarray([target_to_index(edge[1]) for edge in edges], dtype=np.int64)
    return senders, receivers


def edge_attr(edges: set[Edge]) -> NDArray[np.int64]:
    return np.array([edge[2] for edge in edges], dtype=np.int64)


def object_list(
    obs_keys: list[GroundValue],
    objects_with_type: Callable[[GroundValue], list[Object]],
) -> list[Object]:
    unique_objects = {obj for key in obs_keys for obj in objects_with_type(key)}
    # sorted_objects = unique_objects
    return [Object("None", "None")] + list(unique_objects)


def generate_bipartite_obs_func(
    cls: type[T],
    action_fluent_type_mask: Callable[[str], tuple[bool, ...]],
    action_fluent_arity_mask: Callable[[str], tuple[bool, ...]],
):
    def f(
        observations: dict[GroundValue, T],
        groundings: list[GroundValue],
        object_nodes: list[Object],
    ) -> T:
        nullary_groundings = [g for g in groundings if arity(g) == 0]
        non_nullary_groundings = {
            g: idx for idx, g in enumerate(g for g in groundings if arity(g) > 0)
        }

        edges = create_edges(non_nullary_groundings.keys())

        object_names = [obj.name for obj in object_nodes]
        object_types = [obj.type for obj in object_nodes]

        factor_node_values = [observations[g] for g in non_nullary_groundings]
        factor_node_predicates = [predicate(g) for g in non_nullary_groundings]

        object_indices = {name: idx for idx, name in enumerate(object_names)}

        senders, receivers = translate_edges(
            lambda x: non_nullary_groundings[x], lambda x: object_indices[x], edges
        )

        edge_attributes = edge_attr(edges)

        global_variables = [predicate(g) for g in nullary_groundings]
        global_variable_values = [observations[g] for g in nullary_groundings]

        action_type_mask = [
            action_fluent_type_mask(obj_type) for obj_type in object_types
        ]
        action_arity_mask = [
            action_fluent_arity_mask(obj_type) for obj_type in object_types
        ]

        if edges:
            assert senders.max() < len(
                factor_node_values
            ), "Senders index out of bounds."
            assert receivers.max() < len(object_types), "Receivers index out of bounds."

        return cls(
            factor_node_predicates,
            factor_node_values,
            object_names,
            object_types,
            senders,
            receivers,
            edge_attributes,
            global_variables,
            global_variable_values,
            action_type_mask,
            action_arity_mask,
            non_nullary_groundings,
            nullary_groundings,
        )

    return f
