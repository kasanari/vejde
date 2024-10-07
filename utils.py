from typing import NamedTuple, TypeVar, Any
import numpy as np


T = TypeVar("T")


Edge = NamedTuple("Edge", [("predicate", str), ("object", str), ("pos", int)])


def predicate(key: str) -> str:
    return key.split("___")[0]


def objects(key: str) -> list[str]:
    split = key.split("___")
    return split[1].split("__") if len(split) > 1 else []


def translate_edges(
    source_symbols: list[str], target_symbols: list[str], edges: set[Edge]
):
    return np.array(
        [(source_symbols.index(key[0]), target_symbols.index(key[1])) for key in edges],
        dtype=np.uint,
    )


def edge_attr(edges: set[Edge]) -> np.ndarray[np.uint, Any]:
    return np.array([key[2] for key in edges], dtype=np.uint)


def create_edges(d: dict[str, Any]) -> set[Edge]:
    edges: set[Edge] = set()
    for key in d:
        for pos, object in enumerate(objects(key)):
            new_key = Edge(key, object, pos)
            edges.add(new_key)
    return edges


def generate_bipartite_obs(
    obs: dict[str, bool],
    groundings: list[str],
    symb_to_idx: dict[str, int],
    variable_ranges: dict[str, str],
) -> tuple[
    np.ndarray[np.bool_, Any],
    np.ndarray[np.int_, Any],
    np.ndarray[np.uint, Any],
    np.ndarray[np.uint, Any],
    np.ndarray[np.bool_, Any],
]:
    edges: set[Edge] = create_edges(obs)
    obs_objects: set[str] = set()

    for key in obs:
        for object in objects(key):
            obs_objects.add(object)

    object_list: list[str] = sorted(obs_objects)

    object_nodes = np.array(
        [symb_to_idx[object] for object in object_list], dtype=np.int32
    )
    fact_node_values = np.array([obs[key] for key in groundings], dtype=np.float32)

    fact_node_predicate = np.array(
        [symb_to_idx[predicate(key)] for key in groundings], dtype=np.int_
    )

    edge_indices = translate_edges(groundings, object_list, edges)

    edge_attributes = edge_attr(edges)

    fact_nodes = np.stack(
        [fact_node_values, fact_node_predicate], axis=1, dtype=np.float_
    )

    assert max(edge_indices[:, 0]) < len(fact_nodes)
    assert max(edge_indices[:, 1]) < len(object_nodes)

    numeric = np.array(
        [
            1 if variable_ranges[predicate(g)] in ["real", "int"] else 0
            for g in groundings
        ],
        dtype=np.bool_,
    )

    return (fact_nodes, object_nodes, edge_indices, edge_attributes, numeric)
