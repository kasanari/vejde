from collections.abc import Callable, Sequence
from functools import partial
from typing import TypeVar

import numpy as np

from regawa.model.base_grounded_model import Grounding
from regawa import BaseModel
from regawa.data import HeteroObsData
from regawa.model import (
    fn_valid_action_fluents_given_arity,
    fn_valid_action_fluents_given_type,
)
from regawa import GroundObs
from .grounding_utils import (
    bool_groundings,
    fn_is_bool,
    fn_is_numeric,
    numeric_groundings,
    fn_objects_with_type,
)
from .gym_utils import idxgraph_to_obsdata
from .types import (
    FactorGraph,
    HeteroGraph,
    IdxFactorGraph,
    Object,
    StackedFactorGraph,
    Variables,
)
from .utils import (
    generate_bipartite_obs_func,
    map_graph_to_idx,
    object_list,
)

V = TypeVar("V", np.float32, np.bool_)


type StrToInt = Callable[[str], int]


def fn_regular_map_graph_to_idx(rel_to_idx: StrToInt, type_to_idx: StrToInt):
    """
    Prepares a function that maps a FactorGraph with string attributes to a FactorGraph with integer attributes.
    """

    def regular_map_graph_to_idx(factorgraph: FactorGraph[V], var_val_dtype: type):
        """
        Maps a FactorGraph with string attributes to a FactorGraph with integer attributes.
        """
        vars = Variables(
            factorgraph.variables,
            factorgraph.variable_values,
            np.ones_like(factorgraph.variable_values, dtype=np.int64),
        )
        global_vars = Variables(
            factorgraph.global_variables,
            factorgraph.global_variable_values,
            np.ones_like(factorgraph.global_variable_values, dtype=np.int64),
        )
        return map_graph_to_idx(
            vars,
            global_vars,
            factorgraph.senders,
            factorgraph.receivers,
            factorgraph.edge_attributes,
            factorgraph.action_type_mask,
            factorgraph.action_arity_mask,
            factorgraph.factor_types,
            rel_to_idx,
            type_to_idx,
            var_val_dtype,
        )

    return regular_map_graph_to_idx


def fn_heterograph_to_heteroobs(
    fn_graph_to_idx: Callable[
        [
            FactorGraph[V] | StackedFactorGraph[V],
            type,
        ],
        IdxFactorGraph[V],
    ],
):
    """
    Returns a function that takes a HeteroGraph and returns a HeteroObsData (for use in GNNs).
    """

    def heterograph_to_heteroobs(heterogenous_graph: HeteroGraph) -> HeteroObsData:
        return HeteroObsData(
            bool=idxgraph_to_obsdata(
                fn_graph_to_idx(
                    heterogenous_graph.boolean,  # type: ignore
                    np.bool_,
                ),
            ),
            float=idxgraph_to_obsdata(
                fn_graph_to_idx(
                    heterogenous_graph.numeric,  # type: ignore
                    np.float32,
                ),
            ),
        )

    return heterograph_to_heteroobs


def filter_none_groundings(rddl_obs: GroundObs) -> GroundObs:
    filtered_groundings = [
        g
        for g in rddl_obs
        if rddl_obs[g] is not None  # type: ignore
    ]

    filtered_obs: GroundObs = {k: rddl_obs[k] for k in filtered_groundings}
    return filtered_obs


T = TypeVar(
    "T",
    FactorGraph[np.bool_],
    StackedFactorGraph[np.bool_],
)


def fn_obsdict_to_graph_boolean(
    model: BaseModel,
    graph_cls: type[T],
):

    generate_bipartite_obs_bool = generate_bipartite_obs_func(
        graph_cls,
        fn_valid_action_fluents_given_type(model),
        fn_valid_action_fluents_given_arity(model),
    )

    b_g = partial(
        bool_groundings,
        is_bool=fn_is_bool(model.fluent_range),
    )

    def obsdict_to_graph(
        rddl_obs: GroundObs, groundings: Sequence[Grounding], object_nodes: list[Object]
    ):
        return generate_bipartite_obs_bool(
            rddl_obs,
            b_g(groundings),
            object_nodes,
        )

    return obsdict_to_graph


S = TypeVar(
    "S",
    FactorGraph[np.float32],
    StackedFactorGraph[np.float32],
)


def fn_obsdict_to_graph_numeric(model: BaseModel, graph_cls: type[S]):

    generate_bipartite_obs_numeric = generate_bipartite_obs_func(
        graph_cls,
        fn_valid_action_fluents_given_type(model),
        fn_valid_action_fluents_given_arity(model),
    )

    n_g = partial(
        numeric_groundings,
        is_numeric=fn_is_numeric(model.fluent_range),
    )

    def obsdict_to_graph(
        rddl_obs: GroundObs, groundings: Sequence[Grounding], object_nodes: list[Object]
    ):
        return generate_bipartite_obs_numeric(
            rddl_obs,
            n_g(groundings),
            object_nodes,
        )

    return obsdict_to_graph


def fn_obsdict_to_graph(
    model: BaseModel,
    bool_graph_cls: type[T],
    numeric_graph_cls: type[S],
):
    """
    Returns a function that takes an observation dictionary of groundings and values, and returns a heterogenous bipartite graph.
    """

    objects_with_type = fn_objects_with_type(model.fluent_param)
    bool_fn = fn_obsdict_to_graph_boolean(model, bool_graph_cls)
    numeric_fn = fn_obsdict_to_graph_numeric(model, numeric_graph_cls)

    def obsdict_to_graph(rddl_obs: GroundObs):
        filtered_groundings = [
            g
            for g in rddl_obs
            if rddl_obs[g] is not None  # type: ignore
        ]

        filtered_obs: GroundObs = {k: rddl_obs[k] for k in filtered_groundings}

        object_nodes = object_list(list(filtered_obs.keys()), objects_with_type)

        bool_g = bool_fn(filtered_obs, filtered_groundings, object_nodes)
        numeric_g = numeric_fn(filtered_obs, filtered_groundings, object_nodes)
        return HeteroGraph(numeric_g, bool_g)

    return obsdict_to_graph
