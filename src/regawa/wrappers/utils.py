from itertools import chain
import random
from typing import Any, TypeVar
from collections.abc import Callable
import numpy as np
import logging
from gymnasium.spaces import Dict

from regawa.model import GroundValue
from regawa.wrappers.grounding_utils import (
    arity,
    create_edges,
    objects,
    objects_with_type,
    predicate,
)
from regawa.wrappers.util_types import (
    Edge,
    FactorGraph,
    IdxFactorGraph,
    Object,
    RenderGraph,
    StackedFactorGraph,
    Variables,
)
from numpy.typing import NDArray
from numpy import asarray


logger = logging.getLogger(__name__)


V = TypeVar("V")


def map_graph_to_idx[V](
    vars: Variables[V],
    global_vars: Variables[V],
    senders: NDArray[np.int64],
    receivers: NDArray[np.int64],
    edge_attributes: list[int],
    action_mask: NDArray[np.int64],
    factor_types: list[str],
    rel_to_idx: Callable[[str], int],
    type_to_idx: Callable[[str], int],
    var_val_dtype: type,
) -> IdxFactorGraph[V]:
    factor_type_idx = [type_to_idx(object) for object in factor_types]
    idx_global_vars = [rel_to_idx(p) for p in global_vars.types]
    idx_vars = [rel_to_idx(p) for p in vars.types]

    arr = asarray
    return IdxFactorGraph(
        Variables(
            arr(idx_vars, dtype=np.int64),
            arr(vars.values, dtype=var_val_dtype),
            arr(vars.lengths),
        ),
        arr(factor_type_idx, dtype=np.int64),
        senders,
        receivers,
        np.asarray(edge_attributes, dtype=np.int64),
        Variables(
            arr(idx_global_vars, dtype=np.int64),
            arr(global_vars.values, dtype=var_val_dtype),
            arr(global_vars.lengths),
        ),
        action_mask,
    )


def create_render_graph(
    bool_g: FactorGraph[bool], numeric_g: FactorGraph[float]
) -> RenderGraph:
    def format_label(key: GroundValue) -> str:
        fluent, *args = key
        return f"{fluent}({', '.join(args)})" if args else fluent

    boolean_labels = [
        f"{format_label(key)}={bool_g.variable_values[idx]}"
        for idx, key in enumerate(bool_g.groundings)
    ]
    numeric_labels = [
        f"{format_label(key)}={numeric_g.variable_values[idx]}"
        for idx, key in enumerate(numeric_g.groundings)
    ]

    labels = boolean_labels + numeric_labels

    factor_labels = [f"{key}" for key in bool_g.factors]

    edge_attributes = bool_g.edge_attributes + numeric_g.edge_attributes

    senders = np.concatenate(
        [bool_g.senders, numeric_g.senders + len(bool_g.variables)]
    )

    receivers = np.concatenate([bool_g.receivers, numeric_g.receivers])

    global_numeric = [
        f"{key}={numeric_g.global_variable_values[idx]}"
        for idx, key in enumerate(numeric_g.global_variables)
    ]
    global_boolean = [
        f"{key}={bool_g.global_variable_values[idx]}"
        for idx, key in enumerate(bool_g.global_variables)
    ]
    global_labels = global_boolean + global_numeric

    return RenderGraph(
        labels, factor_labels, senders, receivers, edge_attributes, global_labels
    )


def from_dict_action(
    action: tuple[str, ...],
    action_to_idx: Callable[[str], int],
    obj_to_idx: Callable[[str], int],
) -> tuple[int, ...]:
    return tuple([action_to_idx(action[0])] + [obj_to_idx(obj) for obj in action[1:]])


def idx_action_to_ground_value(
    action: tuple[int, ...],
    idx_to_action: Callable[[int], str],
    idx_to_obj: Callable[[int], str],
) -> GroundValue:
    return (idx_to_action(action[0]),) + tuple(
        [idx_to_obj(obj) for obj in action[1:] if obj != 0]
    )


def sample_action(action_space: Dict) -> dict[str, int]:
    action = action_space.sample()  # type: ignore

    action: tuple[str, int] = random.choice(list(action.items()))  # type: ignore
    a = {action[0]: action[1]}
    return a


def translate_edges(
    source_symbols: list[GroundValue], target_symbols: list[str], edges: list[Edge]
):
    return (
        np.array(
            [source_symbols.index(key[0]) for key in edges],
            dtype=np.int64,
        ),
        np.array([target_symbols.index(key[1]) for key in edges], dtype=np.int64),
    )


def edge_attr(edges: set[Edge]) -> np.ndarray[np.uint, Any]:
    return np.array([key[2] for key in edges], dtype=np.int64)


def object_list(
    keys: list[GroundValue], relation_to_types: Callable[[str, int], str]
) -> list[Object]:
    obs_objects: set[Object] = set()
    for key in keys:
        for object in objects_with_type(key, relation_to_types):
            obs_objects.add(object)
    object_list = sorted(obs_objects)
    object_list = [Object("None", "None")] + object_list
    return object_list


def object_list_from_groundings(groundings: list[GroundValue]) -> list[str]:
    return sorted(set(chain(*[objects(g) for g in groundings])))


def predicate_list_from_groundings(groundings: list[GroundValue]) -> list[str]:
    return sorted(set(chain(*[predicate(g) for g in groundings])))


T = TypeVar(
    "T",
    type[FactorGraph[bool]],
    type[StackedFactorGraph[bool]],
    type[FactorGraph[float]],
    type[StackedFactorGraph[float]],
)


def generate_bipartite_obs(
    cls: T,
    obs: dict[GroundValue, bool | float],
    groundings: list[GroundValue],
    relation_to_types: Callable[[str, int], str],
    action_fluent_mask: Callable[[str], tuple[bool, ...]],
    num_actions: int,
    # variable_ranges: dict[str, str],
) -> T:
    nullary_groundings = [g for g in groundings if arity(g) == 0]
    non_nullary_groundings = [g for g in groundings if arity(g) > 0]

    edges = create_edges(non_nullary_groundings)

    obj_list = object_list(
        obs.keys(), relation_to_types
    )  # NOTE: this makes factors common between boolean and numeric

    object_types = [object.type for object in obj_list]

    fact_node_values = [obs[key] for key in non_nullary_groundings]

    fact_node_predicate = [predicate(key) for key in non_nullary_groundings]

    obj_names = [object.name for object in obj_list]

    senders, receivers = translate_edges(
        non_nullary_groundings, obj_names, edges
    )  # edges are (var, factor)

    edge_attributes = [key[2] for key in edges]

    global_variables = [predicate(key) for key in nullary_groundings]
    global_variable_values = [obs[key] for key in nullary_groundings]

    action_mask = [action_fluent_mask(o) for o in object_types]

    if edges:
        assert max(senders) < len(fact_node_values)
        assert max(senders) < len(fact_node_predicate)
        assert max(receivers) < len(object_types)

    return cls(
        fact_node_predicate,
        fact_node_values,  # type: ignore
        obj_names,
        object_types,
        senders,
        receivers,
        edge_attributes,
        global_variables,
        global_variable_values,  # type: ignore
        action_mask,
        non_nullary_groundings,
        nullary_groundings,
    )
