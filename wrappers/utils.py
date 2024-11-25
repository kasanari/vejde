import random
from typing import Any, NamedTuple, TypeVar

import numpy as np
from gymnasium import spaces

T = TypeVar("T")


Edge = NamedTuple("Edge", [("predicate", str), ("object", str), ("pos", int)])

IdxFactorGraph = NamedTuple(
    "IdxFactorGraph",
    [
        ("variables", np.ndarray[np.int64, Any]),
        ("values", np.ndarray[np.bool_, Any]),
        ("factors", np.ndarray[np.int64, Any]),
        ("edge_indices", np.ndarray[np.int64, Any]),
        ("edge_attributes", np.ndarray[np.int64, Any]),
    ],
)

FactorGraph = NamedTuple(
    "FactorGraph",
    [
        ("variables", list[str]),
        ("values", list[bool]),
        ("factors", list[str]),
        ("factor_values", list[str]),
        ("edge_indices", np.ndarray[np.int64, Any]),
        ("edge_attributes", list[str]),
    ],
)


def sample_action(action_space) -> dict[str, int]:
    action: dict[str, int] = action_space.sample()

    action: tuple[str, int] = random.choice(list(action.items()))
    action = {action[0]: action[1]}
    return action


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


def create_edges(d: dict[str, Any]) -> set[Edge]:
    edges: set[Edge] = set()
    for key in d:
        for pos, object in enumerate(objects(key)):
            new_key = Edge(key, object, pos)
            edges.add(new_key)
    return edges


def map_graph_to_idx(
    factorgraph: FactorGraph,
    rel_to_idx: dict[str, int],
    type_to_idx: dict[str, int],
) -> IdxFactorGraph:
    edge_attributes = np.asarray(factorgraph.edge_attributes, dtype=np.int64)

    return IdxFactorGraph(
        np.array([rel_to_idx[key] for key in factorgraph.variables], dtype=np.int64),
        np.array(factorgraph.values, dtype=np.bool_),
        np.array(
            [type_to_idx[object] for object in factorgraph.factor_values],
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


def generate_bipartite_obs(
    obs: dict[str, bool],
    groundings: list[str],
    obj_to_type: dict[str, str],
    variable_ranges: dict[str, str],
) -> FactorGraph:
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

    numeric = np.array(
        [
            1 if variable_ranges[predicate(g)] in ["real", "int"] else 0
            for g in groundings
        ],
        dtype=np.bool_,
    )

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
    predicate_node_classes,
    predicate_node_values,
    object_nodes,
    edges,
    edge_attributes,
    obj_to_idx,
    rel_to_idx,
    numeric,
):
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta"]
    graph = "graph G {\n"
    first_mapping = {}
    second_mapping = {}
    global_idx = 0
    for idx, n_class in enumerate(predicate_node_classes):
        label = (
            f'"{rel_to_idx[int(n_class)]}={predicate_node_values[idx]}"'
            if numeric[idx]
            else f'"{rel_to_idx[int(n_class)]}={bool(predicate_node_values[idx])}"'
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
    predicate_node_values: list[bool],
    object_nodes: list[int],
    edges: list[tuple[int, int]],
    edge_attributes: list[int],
    idx_to_type: dict[int, str],
    idx_to_rel: dict[int, str],
):
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
