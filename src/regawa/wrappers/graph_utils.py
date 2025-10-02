from collections.abc import Callable, Sequence
from functools import partial
from typing import TypeVar

import numpy as np

from regawa.data.graph import GraphTypes, VariableDomain
from regawa.model.base_grounded_model import Grounding
from regawa import BaseModel
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
from regawa.data import (
    HeteroGraph,
    Object,
    StackedStringFactorGraph,
    StringFactorGraph,
    HeteroObsData,
    ObsData,
)
from .utils import (
    generate_bipartite_obs_func,
    object_list,
)


type StrToInt = Callable[[str], int]




def fn_heterograph_to_heteroobs(
    fn_graph_to_idx: Callable[
        [
            GraphTypes,
            type,
        ],
        ObsData[VariableDomain],
    ],
):
    """
    Returns a function that takes a HeteroGraph and returns a HeteroObsData (for use in GNNs).
    """

    def heterograph_to_heteroobs(heterogenous_graph: HeteroGraph) -> HeteroObsData:
        return HeteroObsData(
            bool=fn_graph_to_idx(
                heterogenous_graph.boolean,  # type: ignore
                np.int8,
            ),
            float=fn_graph_to_idx(
                heterogenous_graph.numeric,  # type: ignore
                np.float32,
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


BooleanGraphTypes = TypeVar(
    "BooleanGraphTypes",
    StringFactorGraph[np.bool_],
    StackedStringFactorGraph[np.bool_],
)


def fn_groundobs_to_graph_boolean(
    model: BaseModel,
    graph_cls: type[BooleanGraphTypes],
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


NumericGraphTypes = TypeVar(
    "NumericGraphTypes",
    StringFactorGraph[np.float32],
    StackedStringFactorGraph[np.float32],
)


def fn_groundobs_to_graph_numeric(model: BaseModel, graph_cls: type[NumericGraphTypes]):
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


def fn_groundobs_to_graph(
    model: BaseModel,
    bool_graph_cls: type[BooleanGraphTypes],
    numeric_graph_cls: type[NumericGraphTypes],
) -> Callable[[GroundObs], HeteroGraph]:
    """
    Returns a function that takes an observation dictionary of groundings and values, and returns a heterogenous bipartite graph.
    """

    objects_with_type = fn_objects_with_type(model.fluent_param)
    bool_fn = fn_groundobs_to_graph_boolean(model, bool_graph_cls)
    numeric_fn = fn_groundobs_to_graph_numeric(model, numeric_graph_cls)

    def obsdict_to_graph(rddl_obs: GroundObs) -> HeteroGraph:
        groundings = list(rddl_obs.keys())  # create an order.
        object_nodes = object_list(groundings, objects_with_type)
        return HeteroGraph(
            numeric_fn(rddl_obs, groundings, object_nodes),
            bool_fn(rddl_obs, groundings, object_nodes),
        )

    return obsdict_to_graph
