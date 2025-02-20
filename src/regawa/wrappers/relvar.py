from collections.abc import Callable
from typing import Any, NamedTuple, TypeVar

import numpy as np

from regawa.model.base_model import BaseModel

from ..model import GroundValue
from .grounding_utils import objects, predicate
from .util_types import FactorGraph
from .utils import translate_edges


class Factor(NamedTuple):
    name: tuple[str, ...]
    type: tuple[str, ...]


class LiftedFactorGraph(NamedTuple):
    variables: list[str]
    factors: list[str]
    edge_indices: np.ndarray[np.int64, Any]  # variable to factor


V = TypeVar("V")


def has_shared_entry(s1: set[Any], s2: set[Any]) -> bool:
    return bool(s1.intersection(s2))


def factor_values(model: BaseModel) -> dict[tuple[str, ...], tuple[str, ...]]:
    params = [model.fluent_params(fluent) for fluent in model.fluents]

    unique_params = sorted(set(filter(lambda x: x, params)))

    # params_dict = {
    #     fluent: model.fluent_params(fluent) for fluent in model.relations
    # }

    common_params = {
        param: [
            fluent for fluent in model.fluents if model.fluent_params(fluent) == param
        ]
        for param in params
        if param != ()
    }

    # edges = [(fluent, param) for fluent, param in params_dict.items()]

    factors: dict[GroundValue, tuple[str, ...]] = {}

    factor_vals = dict()

    # match params against each other to find common params
    for f1 in unique_params:
        for f2 in unique_params:
            if has_shared_entry(set(f1), set(f2)):
                if f1 not in factors:
                    factors[f1] = []
                factors[f1].append(f2)

    for f, r in factors.items():
        variables_for_params = set([fluent for p in r for fluent in common_params[p]])
        factor_vals[f] = tuple(variables_for_params)

    return factor_vals


def factor_list(
    groundings: list[GroundValue], relation_to_types: Callable[[str], tuple[str, ...]]
) -> list[Factor]:
    obs_objects: set[Factor] = set()
    for key in groundings:
        for object in factors_with_type(key, relation_to_types):
            obs_objects.add(object)
    object_list = sorted(obs_objects)
    # object_list = [Factor(("None",), ("None",))] + object_list
    return object_list


def factors_with_type(
    key: GroundValue, relation_to_types: Callable[[str], tuple[str, ...]]
) -> list[Factor]:
    p = predicate(key)
    os = objects(key)
    return [Factor(os, relation_to_types(p))]


def relvar_edges(
    groundings: list[GroundValue],
    fluent_params: Callable[[str], tuple[str, ...]],
    factor_vals: Callable[[tuple[str, ...]], tuple[str, ...]],
):
    factors = factor_list(groundings, fluent_params)
    edges = set()
    for f in factors:
        related_variables = factor_vals(f.type)
        for v in related_variables:
            for k in filter(lambda x: predicate(x) == v, groundings):
                args = objects(k)
                if has_shared_entry(set(args), set(f.name)):
                    edges.add((k, f.name))
    return sorted(edges)


def relvar_obs(
    obs: dict[str, V],
    groundings: list[GroundValue],
    fluent_params: Callable[[str], tuple[str, ...]],
    factor_vals: Callable[[tuple[str, ...]], tuple[str, ...]],
) -> FactorGraph[V]:
    edges = relvar_edges(groundings, fluent_params, factor_vals)
    factors = factor_list(groundings, fluent_params)

    variable_values = [obs[key] for key in groundings]
    variable_types = [predicate(key) for key in groundings]

    factor_types = [f.type for f in factors]
    factor_names = [f.name for f in factors]

    senders, receivers = translate_edges(
        groundings, factor_names, edges
    )  # edges are (var, factor)

    return FactorGraph(
        variable_types,
        variable_values,
        factor_names,
        factor_types,
        senders=senders,
        receivers=receivers,
        edge_attributes=[1 for _ in edges],
        global_variable_values=None,
        global_variables=None,
        action_mask=None,
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
    for idx, variable_type in enumerate(fg.variables):
        label = (
            # f'"{rel_to_idx[int(n_class)]}={predicate_node_values[idx]}"'
            # if numeric[idx]
            f'"{variable_type}={bool(fg.variable_values[idx])}"'
        )
        graph += f'"{global_idx}" [label={label}]\n'
        first_mapping[idx] = global_idx
        global_idx += 1
    for idx, factor_type in enumerate(fg.factors):
        graph += f'"{global_idx}" [label="{factor_type}", shape=box]\n'
        second_mapping[idx] = global_idx
        global_idx += 1
    for attribute, edge in zip(fg.edge_attributes, fg.edge_indices):
        graph += f'"{first_mapping[edge[0]]}" -- "{second_mapping[edge[1]]}" [color="{colors[attribute]}"]\n'
    graph += "}"
    return graph


def lifted_to_graphviz(
    fg: LiftedFactorGraph,
    # numeric,
):
    graph = "graph G {\n"
    graph += "overlap_scaling=-20\n"
    first_mapping = {}
    second_mapping = {}
    global_idx = 0
    for idx, variable_type in enumerate(fg.variables):
        label = f'"{variable_type}"'
        graph += f'"{global_idx}" [label={label}]\n'
        first_mapping[idx] = global_idx
        global_idx += 1
    for idx, factor_type in enumerate(fg.factors):
        graph += f'"{global_idx}" [label="{factor_type}", shape=box]\n'
        second_mapping[idx] = global_idx
        global_idx += 1
    for edge in fg.edge_indices:
        graph += f'"{first_mapping[edge[0]]}" -- "{second_mapping[edge[1]]}"\n'
    graph += "}"
    return graph
