from collections import deque
from itertools import chain
import random
from typing import Any, NamedTuple, TypeVar
from collections.abc import Callable
from pyRDDLGym.core.compiler.model import RDDLLiftedModel  # type: ignore
import numpy as np
import logging
from gymnasium.spaces import Dict

logger = logging.getLogger(__name__)

T = TypeVar("T")


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
    variable_values: list[bool]
    factors: list[str]
    factor_values: list[str]
    edge_indices: np.ndarray[np.int64, Any]
    edge_attributes: list[int]


class StackedFactorGraph(NamedTuple):
    variables: list[str]
    variable_values: list[list[bool]]
    factors: list[str]
    factor_values: list[str]
    edge_indices: np.ndarray[np.int64, Any]
    edge_attributes: list[int]


def get_groundings(model: RDDLLiftedModel, fluents: dict[str, Any]) -> set[str]:
    return set(
        dict(model.ground_vars_with_values(fluents)).keys()  # type: ignore
    )


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
    non_fluent_values: dict[str, Any],
    rel_to_idx: dict[str, int],
    type_to_idx: dict[str, int],
    groundings: list[str],
    obj_to_type: dict[str, str],
    variable_ranges: dict[str, str],
    skip_fluent: Callable[[str, dict[str, str]], bool],
) -> tuple[dict[str, Any], FactorGraph]:
    rddl_obs |= non_fluent_values

    filtered_groundings = sorted(
        [g for g in groundings if not skip_fluent(g, variable_ranges) and g in rddl_obs]
    )

    filtered_obs: dict[str, Any] = {k: rddl_obs[k] for k in filtered_groundings}

    g = generate_bipartite_obs(
        filtered_obs,
        filtered_groundings,
        obj_to_type,
    )

    obs = graph_to_dict(map_graph_to_idx(g, rel_to_idx, type_to_idx))

    return obs, g


def create_stacked_obs(
    rddl_obs: dict[str, Any],
    non_fluent_values: dict[str, Any],
    rel_to_idx: dict[str, int],
    type_to_idx: dict[str, int],
    groundings: list[str],
    obj_to_type: dict[str, str],
    variable_ranges: dict[str, str],
    skip_fluent: Callable[[str, dict[str, str]], bool],
) -> tuple[dict[str, Any], StackedFactorGraph]:
    rddl_obs |= non_fluent_values

    filtered_groundings = sorted(
        [g for g in groundings if not skip_fluent(g, variable_ranges) and g in rddl_obs]
    )

    filtered_obs: dict[str, Any] = {k: rddl_obs[k] for k in filtered_groundings}

    g = generate_bipartite_obs(
        filtered_obs,
        filtered_groundings,
        obj_to_type,
    )

    assert isinstance(g, StackedFactorGraph)

    obs = graph_to_dict(map_stacked_graph_to_idx(g, rel_to_idx, type_to_idx))

    return obs, g


def to_rddl_action(
    action: tuple[int, int],
    idx_to_action: list[str],
    idx_to_obj: list[str],
    action_groundings: set[str],
) -> tuple[dict[str, int], str]:
    action_fluent = idx_to_action[action[0]]
    object_id = idx_to_obj[action[1]]

    rddl_action = (
        f"{action_fluent}___{object_id}" if action_fluent != "noop" else "noop"
    )

    invalid_action = rddl_action not in action_groundings

    if invalid_action:
        logger.warning(f"Invalid action: {rddl_action}")

    rddl_action_dict = (
        {} if invalid_action or action_fluent == "noop" else {rddl_action: 1}
    )
    return rddl_action_dict, rddl_action


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


def arity(grounding: str):
    o = objects(grounding)
    return len(o)


def translate_edges(
    source_symbols: list[str], target_symbols: list[str], edges: set[Edge]
):
    return np.array(
        [(source_symbols.index(key[0]), target_symbols.index(key[1])) for key in edges],
        dtype=np.int64,
    )


def edge_attr(edges: set[Edge]) -> np.ndarray[np.uint, Any]:
    return np.array([key[2] for key in edges], dtype=np.int64)


def create_edges(d: dict[str, Any]) -> list[Edge]:
    edges: deque[Edge] = deque()
    keys = sorted(d.keys())
    for key in keys:
        for pos, object in enumerate(objects(key)):
            new_key = Edge(key, object, pos)
            edges.append(new_key)
    return list(edges)


def map_graph_to_idx(
    factorgraph: FactorGraph,
    rel_to_idx: dict[str, int],
    type_to_idx: dict[str, int],
) -> IdxFactorGraph:
    edge_attributes = np.asarray(factorgraph.edge_attributes, dtype=np.int64)

    vals = np.array(
        factorgraph.variable_values, dtype=np.bool_
    )  # TODO: handle None values in a better way

    return IdxFactorGraph(
        np.array([rel_to_idx[key] for key in factorgraph.variables], dtype=np.int64),
        np.array(vals, dtype=np.int8),
        np.array(
            [type_to_idx[object] for object in factorgraph.factor_values],
            dtype=np.int64,
        ),
        factorgraph.edge_indices,
        edge_attributes,
    )


def map_stacked_graph_to_idx(
    factorgraph: StackedFactorGraph,
    rel_to_idx: dict[str, int],
    type_to_idx: dict[str, int],
) -> IdxFactorGraph:
    edge_attributes = np.asarray(factorgraph.edge_attributes, dtype=np.int64)

    vals = list(chain(*factorgraph.variable_values))

    vars = [
        [factorgraph.variables[i] for _ in v]
        for i, v in enumerate(factorgraph.variable_values)
    ]
    vars = list(chain(*vars))

    vars = [rel_to_idx[p] for p in vars]

    factors = [type_to_idx[object] for object in factorgraph.factor_values]

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


def object_list(obs: dict[str, Any]) -> list[str]:
    obs_objects: set[str] = set()
    for key in obs:
        for object in objects(key):
            obs_objects.add(object)
    object_list: list[str] = sorted(obs_objects)
    return object_list


def object_list_from_groundings(groundings: list[str]) -> list[str]:
    return sorted(set(chain(*[objects(g) for g in groundings])))


def predicate_list_from_groundings(groundings: list[str]) -> list[str]:
    return sorted(set(chain(*[predicate(g) for g in groundings])))


def generate_bipartite_obs(
    obs: dict[str, bool],
    groundings: list[str],
    obj_to_type: dict[str, str],
    # variable_ranges: dict[str, str],
) -> FactorGraph | StackedFactorGraph:
    edges: set[Edge] = create_edges(obs)

    obj_list = object_list(obs)

    object_types = [obj_to_type[object] for object in obj_list]

    fact_node_values = [obs[key] for key in groundings]

    fact_node_predicate = [predicate(key) for key in groundings]

    edge_indices = translate_edges(
        groundings, obj_list, edges
    )  # edges are (var, factor)

    edge_attributes = [key[2] for key in edges]

    assert max(edge_indices[:, 0]) < len(fact_node_values)
    assert max(edge_indices[:, 0]) < len(fact_node_predicate)
    assert max(edge_indices[:, 1]) < len(object_types)

    # numeric = np.array(
    #     [
    #         1 if variable_ranges[predicate(g)] in ["real", "int"] else 0
    #         for g in groundings
    #     ],
    #     dtype=np.bool_,
    # )

    return FactorGraph(
        fact_node_predicate,
        fact_node_values,
        obj_list,
        object_types,
        edge_indices,
        edge_attributes,
        # numeric,
    )


def to_graphviz(
    predicate_node_classes: list[str],
    predicate_node_values: list[int],
    object_nodes: list[str],
    edges: list[tuple[int, int]],
    edge_attributes: list[int],
    obj_to_idx: dict[str, int],
    rel_to_idx: dict[str, int],
    # numeric,
):
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta"]
    graph = "graph G {\n"
    first_mapping = {}
    second_mapping = {}
    global_idx = 0
    for idx, n_class in enumerate(predicate_node_classes):
        label = (
            # f'"{rel_to_idx[int(n_class)]}={predicate_node_values[idx]}"'
            # if numeric[idx]
            f'"{rel_to_idx[n_class]}={bool(predicate_node_values[idx])}"'
        )
        graph += f'"{global_idx}" [label={label}, shape=box]\n'
        first_mapping[idx] = global_idx
        global_idx += 1
    for idx, data in enumerate(object_nodes):
        graph += f'"{global_idx}" [label="{obj_to_idx[data]}", shape=circle]\n'
        second_mapping[idx] = global_idx
        global_idx += 1
    for attribute, edge in zip(edge_attributes, edges):
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
    first_mapping = {}
    second_mapping = {}
    global_idx = 0
    for idx, n_class in enumerate(predicate_node_classes):
        label = f'"{idx_to_rel[int(n_class)]}={bool(predicate_node_values[idx])}"'
        graph += f'"{global_idx}" [label={label}, shape=box]\n'
        first_mapping[idx] = global_idx
        global_idx += 1
    for idx, data in enumerate(object_nodes):
        graph += f'"{global_idx}" [label="{idx_to_type[data]}", shape=circle]\n'
        second_mapping[idx] = global_idx
        global_idx += 1
    for attribute, edge in zip(edge_attributes, edges):
        graph += f'"{first_mapping[edge[0]]}" -- "{second_mapping[edge[1]]}" [color="{colors[attribute]}"]\n'
    graph += "}"
    return graph
