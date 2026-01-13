from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import cache, partial
import itertools
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from .heterograph import HeteroGraph


from .actions import ActionMask, fn_action_masks
from .graph import (
    Edge,
    NullObject,
    Object,
    StringVariables,
    Variables,
)
from .graph import Factors
from .stacked_graph import StackedStringFactorGraph
from regawa.model.grounding_func import bool_groundings, fn_is_bool, objects, predicate
from regawa.model import Grounding
from regawa import BaseModel
from regawa import GroundObs
from regawa.model.grounding_func import (
    arity,
    fn_is_numeric,
    numeric_groundings,
)
from .graph import Edges, StringFactorGraph, StringFactors, VariableDomain


type StrToInt = Callable[[str], int]


GraphTypes = TypeVar(
    "GraphTypes",
    StringFactorGraph[np.bool_],
    StackedStringFactorGraph[np.bool_],
    StringFactorGraph[np.float32],
    StackedStringFactorGraph[np.float32],
)

BooleanGraphTypes = TypeVar(
    "BooleanGraphTypes",
    StringFactorGraph[np.bool_],
    StackedStringFactorGraph[np.bool_],
)


NumericGraphTypes = TypeVar(
    "NumericGraphTypes",
    StringFactorGraph[np.float32],
    StackedStringFactorGraph[np.float32],
)


def filter_none_groundings(rddl_obs: GroundObs) -> GroundObs:
    filtered_groundings = [
        g
        for g in rddl_obs
        if rddl_obs[g] is not None  # type: ignore
    ]

    filtered_obs: GroundObs = {k: rddl_obs[k] for k in filtered_groundings}
    return filtered_obs


def fn_groundobs_to_graph_numeric(model: BaseModel, graph_cls: type[NumericGraphTypes]):
    generate_bipartite_obs_numeric = generate_bipartite_obs_func(
        graph_cls, fn_action_masks(model)
    )

    n_g = partial(
        numeric_groundings,
        is_numeric=fn_is_numeric(model.fluent_range),
    )

    def obsdict_to_graph(
        rddl_obs: GroundObs,
        groundings: Sequence[Grounding],
        object_nodes: Sequence[Object],
    ):
        return generate_bipartite_obs_numeric(
            rddl_obs,  # type: ignore
            n_g(groundings),
            object_nodes,
        )

    return obsdict_to_graph


def fn_objects_with_type(relation_to_types: Callable[[str, int], str]):
    """
    Returns a function that takes a grounding and returns a list of Objects (object name, type).
    """

    @cache
    def objects_with_type(
        key: Grounding,
    ) -> list[Object]:
        p = predicate(key)
        os = objects(key)
        return [Object(o, relation_to_types(p, i)) for i, o in enumerate(os)]

    return objects_with_type


def object_list(
    obs_keys: Sequence[Grounding],
    objects_with_type: Callable[[Grounding], Sequence[Object]],
) -> Sequence[Object]:
    unique_objects = {obj for key in obs_keys for obj in objects_with_type(key)}
    # sorted_objects = unique_objects
    return [NullObject] + list(unique_objects)


def fn_groundobs_to_heterograph(
    model: BaseModel,
    stacking: bool = False,
) -> Callable[[GroundObs], HeteroGraph]:
    """
    Returns a function that takes an observation dictionary of groundings and values, and returns a heterogenous bipartite graph.
    """

    bool_graph_cls, numeric_graph_cls = (
        (StackedStringFactorGraph[np.bool_], StackedStringFactorGraph[np.float32])
        if stacking
        else (StringFactorGraph[np.bool_], StringFactorGraph[np.float32])
    )

    objects_with_type = fn_objects_with_type(model.fluent_param)
    bool_fn = fn_groundobs_to_graph_boolean(model, bool_graph_cls)  # type: ignore
    numeric_fn = fn_groundobs_to_graph_numeric(model, numeric_graph_cls)  # type: ignore

    def obsdict_to_graph(rddl_obs: GroundObs) -> HeteroGraph:
        groundings = list(rddl_obs.keys())  # create an order.
        object_nodes = object_list(groundings, objects_with_type)
        return HeteroGraph(
            numeric_fn(rddl_obs, groundings, object_nodes),
            bool_fn(rddl_obs, groundings, object_nodes),
        )

    return obsdict_to_graph


def fn_groundobs_to_graph_boolean(
    model: BaseModel,
    graph_cls: type[BooleanGraphTypes],
):
    generate_bipartite_obs_bool = generate_bipartite_obs_func(
        graph_cls,
        fn_action_masks(model),
    )

    b_g = partial(
        bool_groundings,
        is_bool=fn_is_bool(model.fluent_range),
    )

    def obsdict_to_graph(
        rddl_obs: GroundObs,
        groundings: Sequence[Grounding],
        object_nodes: Sequence[Object],
    ):
        return generate_bipartite_obs_bool(
            rddl_obs,  # type: ignore
            b_g(groundings),
            object_nodes,
        )

    return obsdict_to_graph


@cache
def get_edges(key: Grounding) -> list[Edge]:
    """
    Returns a list of edges for a given grounding.
    Each edge connects the predicate to one of its objects.
    An edge is represented as a tuple (predicate, object, position).
    """
    return [Edge(key, object, pos) for pos, object in enumerate(objects(key))]


def create_edges(d: Iterable[Grounding]) -> list[Edge]:
    edges = [get_edges(key) for key in d]
    return list(itertools.chain(*edges))


def translate_edges(
    source_to_index: Callable[[Grounding], int],
    target_to_index: Callable[[str], int],
    edges: list[Edge],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    senders = np.asarray([source_to_index(edge[0]) for edge in edges], dtype=np.int64)
    receivers = np.asarray([target_to_index(edge[1]) for edge in edges], dtype=np.int64)
    return senders, receivers


def factor_to_idx(type_to_idx: Callable[[str], int]):
    def map_factors_to_idx(
        factors: StringFactors,
    ) -> Factors:
        arr = np.asarray
        factor_type_idx = arr(
            [type_to_idx(f_type) for f_type in factors.types], dtype=np.int64
        )
        return Factors(
            factor_type_idx,
            factor_type_idx.shape[0],  # number of factors
        )

    return map_factors_to_idx


def fn_variables_to_idx_with_time(
    rel_to_idx: Callable[[str], int],
):
    def map_variables_to_idx(
        variables: StringVariables[VariableDomain], var_val_dtype: type
    ) -> Variables[VariableDomain]:
        arr = np.asarray

        times, variable_values = (  # type: ignore
            zip(*variables.values) if variables.values else ([], [])
        )

        return Variables(
            arr([rel_to_idx(p) for p in variables.types], dtype=np.int64),
            arr(variable_values, dtype=var_val_dtype),
            arr(variables.length),
            n_variable=variables.n_variable,
            times=arr(times, dtype=np.int64),
        )

    return map_variables_to_idx


def create_variables(
    observations: Mapping[Grounding, VariableDomain],
    groundings: Sequence[Grounding],
) -> StringVariables[VariableDomain]:
    factor_node_values = [observations[g] for g in groundings]
    lengths = [len(x) if isinstance(x, Sequence) else 1 for x in factor_node_values]
    factor_node_predicates = [predicate(g) for g in groundings]
    return StringVariables[VariableDomain](  # type: ignore
        factor_node_predicates,
        factor_node_values,
        lengths,
        len(groundings),
        groundings,
    )


def edge_attr(edges: Iterable[Edge]) -> NDArray[np.int64]:
    return np.asarray([edge[2] for edge in edges], dtype=np.int64)


def fn_variables_to_idx(
    rel_to_idx: Callable[[str], int],
):
    def map_variables_to_idx(
        variables: StringVariables[VariableDomain], var_val_dtype: type
    ) -> Variables[VariableDomain]:
        arr = np.asarray
        return Variables(
            arr([rel_to_idx(p) for p in variables.types], dtype=np.int64),
            arr(variables.values, dtype=var_val_dtype),
            arr(variables.length),
            n_variable=variables.n_variable,
            times=np.zeros((variables.n_variable, 2), dtype=np.int64),
        )

    return map_variables_to_idx


def generate_bipartite_obs_func(
    cls: type[GraphTypes],
    action_mask_func: Callable[[Sequence[str]], ActionMask],
):
    def f(
        observations: Mapping[Grounding, VariableDomain],
        groundings: Sequence[Grounding],
        object_nodes: Sequence[Object],
    ) -> GraphTypes:
        nullary_groundings = [g for g in groundings if arity(g) == 0]
        non_nullary_groundings = {
            g: idx for idx, g in enumerate(g for g in groundings if arity(g) > 0)
        }

        object_names = [obj.name for obj in object_nodes]
        object_types = [obj.type for obj in object_nodes]
        object_indices = {o.name: idx for idx, o in enumerate(object_nodes)}

        edges = create_edges(non_nullary_groundings.keys())
        v_to_f, f_to_v = translate_edges(
            lambda x: non_nullary_groundings[x], lambda x: object_indices[x], edges
        )

        g = cls(
            create_variables(observations, non_nullary_groundings.keys()),  # type: ignore
            StringFactors(
                object_names,
                object_types,
            ),
            Edges(v_to_f, f_to_v, edge_attr(edges)),
            create_variables(observations, nullary_groundings),  # type: ignore
            action_mask_func(object_types),
        )

        if edges:
            assert v_to_f.max() < len(
                g.variables.values
            ), "Senders index out of bounds."
            assert f_to_v.max() < len(object_types), "Receivers index out of bounds."

        return g

    return f
