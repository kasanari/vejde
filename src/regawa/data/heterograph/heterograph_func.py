from collections.abc import Callable, Sequence
from functools import partial

import numpy as np
from pyparsing import TypeVar

from regawa.data.factor_graph import StringFactorGraph
from regawa.data.func import generate_bipartite_obs_func
from regawa.data.graph import (
    Object,
    fn_action_masks,
    fn_objects_with_type,
    object_list,
)
from regawa.data.stacked import StackedStringFactorGraph
from regawa.model import (
    BaseModel,
    Grounding,
    GroundObs,
    bool_groundings,
    fn_is_bool,
    fn_is_numeric,
    numeric_groundings,
)

from .heterograph import HeteroGraph

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
