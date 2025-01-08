from collections import deque
from itertools import chain
import random
from typing import Any, NamedTuple, TypeVar
from collections.abc import Callable
import numpy as np
import logging
from gymnasium.spaces import Dict

from model.base_model import BaseModel

logger = logging.getLogger(__name__)


class Object(NamedTuple):
    name: str
    type: str


class Edge(NamedTuple):
    predicate: str
    object: str
    pos: int


class IdxFactorGraph(NamedTuple):
    variables: np.ndarray[np.int64, Any]
    values: np.ndarray[np.int64, Any]
    factors: np.ndarray[np.int64, Any]
    edge_indices: np.ndarray[np.int64, Any]
    edge_attributes: np.ndarray[np.int64, Any]


class FactorGraph(NamedTuple):
    variables: list[str]
    variable_values: list[bool | float]
    factors: list[str]
    factor_values: list[str]
    edge_indices: np.ndarray[np.int64, Any]  # variable to factor
    edge_attributes: list[int]


class StackedFactorGraph(NamedTuple):
    variables: list[str]
    variable_values: list[list[bool]]
    factors: list[str]
    factor_values: list[str]
    edge_indices: np.ndarray[np.int64, Any]
    edge_attributes: list[int]


def graph_to_dict(idx_g: IdxFactorGraph) -> dict[str, Any]:
    return {
        "var_type": idx_g.variables,
        "var_value": idx_g.values,
        "factor": idx_g.factors,
        "edge_index": idx_g.edge_indices.T,
        "edge_attr": idx_g.edge_attributes,
        # "numeric": numeric,
    }


def create_obs(
    rddl_obs: dict[str, Any],
    model: BaseModel,
    skip_fluent: Callable[[str, dict[str, str]], bool],
) -> tuple[dict[str, Any], FactorGraph, list[str]]:
    filtered_groundings = sorted(
        [
            g
            for g in rddl_obs
            if not skip_fluent(g, model.fluent_range)  # type: ignore
        ]
    )

    filtered_obs: dict[str, Any] = {k: rddl_obs[k] for k in filtered_groundings}

    boolean_groundings = [
        g for g in filtered_groundings if model.fluent_range(predicate(g)) is bool
    ]

    bool_g = generate_bipartite_obs(
        FactorGraph,
        filtered_obs,
        boolean_groundings,
        model.fluent_param,
    )

    numeric_groundings = [
        g
        for g in filtered_groundings
        if (
            model.fluent_range(predicate(g)) is float
            or model.fluent_range(predicate(g)) is int
        )
    ]

    if numeric_groundings:
        _numeric_g = generate_bipartite_obs(  # TODO add this to obs
            FactorGraph,
            filtered_obs,
            numeric_groundings,
            model.fluent_param,
        )

    assert isinstance(bool_g, FactorGraph)

    obs = graph_to_dict(
        map_graph_to_idx(bool_g, model.fluent_to_idx, model.type_to_idx)
    )

    return obs, bool_g, filtered_groundings


def create_stacked_obs(
    rddl_obs: dict[str, Any],
    model: BaseModel,
    skip_fluent: Callable[[str, dict[str, str]], bool],
) -> tuple[dict[str, Any], StackedFactorGraph, list[str]]:
    filtered_groundings = sorted(
        [
            g
            for g in rddl_obs
            if not skip_fluent(g, model.fluent_range)  # type: ignore
        ]
    )

    filtered_obs: dict[str, Any] = {k: rddl_obs[k] for k in filtered_groundings}

    boolean_groundings = [
        g for g in filtered_groundings if model.fluent_range(predicate(g)) is bool
    ]

    bool_g = generate_bipartite_obs(
        StackedFactorGraph,
        filtered_obs,
        boolean_groundings,
        model.fluent_param,
    )

    numeric_groundings = [
        g
        for g in filtered_groundings
        if (
            model.fluent_range(predicate(g)) is float
            or model.fluent_range(predicate(g)) is int
        )
    ]

    if numeric_groundings:
        _numeric_g = generate_bipartite_obs(  # TODO add this to obs
            StackedFactorGraph,
            filtered_obs,
            numeric_groundings,
            model.fluent_param,
        )

    assert isinstance(bool_g, StackedFactorGraph)

    obs = graph_to_dict(
        map_stacked_graph_to_idx(bool_g, model.fluent_to_idx, model.type_to_idx)
    )

    return obs, bool_g, boolean_groundings


def num_edges(arities: Callable[[str], int], groundings: list[str]) -> int:
    return sum(arities(predicate(g)) for g in groundings)


def to_dict_action(
    action: tuple[int, int],
    idx_to_action: Callable[[int], str],
    idx_to_type: Callable[[int], str],
    idx_to_obj: Callable[[int], str],
    fluent_params: Callable[[str], list[str]],
) -> tuple[dict[str, int], str]:
    action_fluent = idx_to_action(action[0])

    param_types = fluent_params(action_fluent)

    grounded_action = f"{action_fluent}"

    params: list[str] = [
        f"{idx_to_obj(action[i + 1])}" for i, _ in enumerate(param_types)
    ]
    for i, intended_param in enumerate(param_types):
        actual_param = idx_to_type(action[i + 1])
        if intended_param != actual_param:
            logger.warning(
                f"Invalid parameter type for fluent {action_fluent} in position {i}. Expected {intended_param} but got {actual_param}."
            )
            action_fluent = "None"
            break

    grounded_action = (
        f"{grounded_action}___{'__'.join(params)}" if params else grounded_action
    )

    action_dict = {} if action_fluent == "None" else {grounded_action: 1}

    return action_dict, grounded_action


def sample_action(action_space: Dict) -> dict[str, int]:
    action = action_space.sample()  # type: ignore

    action: tuple[str, int] = random.choice(list(action.items()))  # type: ignore
    a = {action[0]: action[1]}
    return a


def predicate(key: str) -> str:
    return key.split("___")[0]


def objects(key: str) -> list[str]:
    split = key.split("___")
    return split[1].split("__") if len(split) > 1 else []


def objects_with_type(
    key: str, relation_to_types: Callable[[str, int], str]
) -> list[Object]:
    p = predicate(key)
    os = objects(key)
    return [Object(o, relation_to_types(p, i)) for i, o in enumerate(os)]


def arity(grounding: str):
    o = objects(grounding)
    return len(o)


def translate_edges(
    source_symbols: list[str], target_symbols: list[str], edges: list[Edge]
):
    return np.array(
        [(source_symbols.index(key[0]), target_symbols.index(key[1])) for key in edges],
        dtype=np.int64,
    )


def edge_attr(edges: set[Edge]) -> np.ndarray[np.uint, Any]:
    return np.array([key[2] for key in edges], dtype=np.int64)


def create_edges(d: list[str]) -> list[Edge]:
    edges: deque[Edge] = deque()
    keys = sorted(d)
    for key in keys:
        for pos, object in enumerate(objects(key)):
            new_key = Edge(key, object, pos)
            edges.append(new_key)
    return list(edges)


def map_graph_to_idx(
    factorgraph: FactorGraph,
    rel_to_idx: Callable[[str], int],
    type_to_idx: Callable[[str], int],
) -> IdxFactorGraph:
    edge_attributes = np.asarray(factorgraph.edge_attributes, dtype=np.int64)

    vals = np.array(
        factorgraph.variable_values, dtype=np.bool_
    )  # TODO: handle None values in a better way

    return IdxFactorGraph(
        np.array([rel_to_idx(key) for key in factorgraph.variables], dtype=np.int64),
        np.array(vals, dtype=np.int8),
        np.array(
            [type_to_idx(object) for object in factorgraph.factor_values],
            dtype=np.int64,
        ),
        factorgraph.edge_indices,
        edge_attributes,
    )


def map_stacked_graph_to_idx(
    factorgraph: StackedFactorGraph,
    rel_to_idx: Callable[[str], int],
    type_to_idx: Callable[[str], int],
) -> IdxFactorGraph:
    edge_attributes = np.asarray(factorgraph.edge_attributes, dtype=np.int64)

    # Flatten the list of node history lists to account for different node history lengths
    vals = list(chain(*factorgraph.variable_values))

    vars = [
        [factorgraph.variables[i] for _ in v]
        for i, v in enumerate(factorgraph.variable_values)
    ]
    vars = list(chain(*vars))

    vars = [rel_to_idx(p) for p in vars]

    factors = [type_to_idx(object) for object in factorgraph.factor_values]

    return IdxFactorGraph(
        np.array(vars, dtype=np.int64),
        np.array(vals, dtype=np.int8),
        np.array(
            factors,
            dtype=np.int64,
        ),
        factorgraph.edge_indices,
        edge_attributes,
    )


def object_list(
    obs: dict[str, Any], relation_to_types: Callable[[str, int], str]
) -> list[Object]:
    obs_objects: set[Object] = set()
    for key in obs:
        for object in objects_with_type(key, relation_to_types):
            obs_objects.add(object)
    object_list = sorted(obs_objects)
    object_list = [Object("None", "None")] + object_list
    return object_list


def object_list_from_groundings(groundings: list[str]) -> list[str]:
    return sorted(set(chain(*[objects(g) for g in groundings])))


def predicate_list_from_groundings(groundings: list[str]) -> list[str]:
    return sorted(set(chain(*[predicate(g) for g in groundings])))


T = TypeVar("T", type[FactorGraph], type[StackedFactorGraph])


def generate_bipartite_obs(
    cls: T,
    obs: dict[str, bool | float],
    groundings: list[str],
    relation_to_types: Callable[[str, int], str],
    # variable_ranges: dict[str, str],
) -> T:
    edges = create_edges(groundings)

    obj_list = object_list(obs, relation_to_types)

    object_types = [object.type for object in obj_list]

    fact_node_values = [obs[key] for key in groundings]

    fact_node_predicate = [predicate(key) for key in groundings]

    obj_names = [object.name for object in obj_list]

    edge_indices = translate_edges(
        groundings, obj_names, edges
    )  # edges are (var, factor)

    edge_attributes = [key[2] for key in edges]

    assert max(edge_indices[:, 0]) < len(fact_node_values)
    assert max(edge_indices[:, 0]) < len(fact_node_predicate)
    assert max(edge_indices[:, 1]) < len(object_types)

    return cls(
        fact_node_predicate,
        fact_node_values,
        obj_names,
        object_types,
        edge_indices,
        edge_attributes,
    )


def to_graphviz(
    fg: FactorGraph,
    # numeric,
):
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta"]
    graph = "graph G {\n"
    graph += "overlap_scaling=-20\n"
    first_mapping = {}
    second_mapping = {}
    global_idx = 0
    for idx, n_class in enumerate(fg.variables):
        label = (
            # f'"{rel_to_idx[int(n_class)]}={predicate_node_values[idx]}"'
            # if numeric[idx]
            f'"{n_class}={bool(fg.variable_values[idx])}"'
        )
        graph += f'"{global_idx}" [label={label}]\n'
        first_mapping[idx] = global_idx
        global_idx += 1
    for idx, data in enumerate(fg.factors):
        graph += f'"{global_idx}" [label="{data}", shape=box]\n'
        second_mapping[idx] = global_idx
        global_idx += 1
    for attribute, edge in zip(fg.edge_attributes, fg.edge_indices):
        graph += f'"{first_mapping[edge[0]]}" -- "{second_mapping[edge[1]]}" [color="{colors[attribute]}"]\n'
    graph += "}"
    return graph


def to_graphviz_alt(
    predicate_node_classes: list[int],
    predicate_node_values: list[int],
    object_nodes: list[int],
    edges: list[tuple[int, int]],
    edge_attributes: list[int],
    idx_to_type: list[str],
    idx_to_rel: list[str],
) -> str:
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta"]
    graph = "graph G {\n"
    graph += "overlap_scaling=-20\n"
    first_mapping = {}
    second_mapping = {}
    global_idx = 0
    for idx, n_class in enumerate(predicate_node_classes):
        label = f'"{idx_to_rel(int(n_class))}={bool(predicate_node_values[idx])}"'
        graph += f'"{global_idx}" [label={label}]\n'
        first_mapping[idx] = global_idx
        global_idx += 1
    for idx, data in enumerate(object_nodes):
        graph += f'"{global_idx}" [label="{idx_to_type(data)}", shape=box]\n'
        second_mapping[idx] = global_idx
        global_idx += 1
    for attribute, edge in zip(edge_attributes, edges):
        graph += f'"{first_mapping[edge[0]]}" -- "{second_mapping[edge[1]]}" [color="{colors[attribute]}"]\n'
    graph += "}"
    return graph
