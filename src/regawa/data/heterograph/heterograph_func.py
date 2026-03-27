from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import TypeVar

import numpy as np

from regawa.data.factor_graph import StringFactorGraph
from regawa.data.graph import (
    ActionMask,
    Edges,
    Object,
    StringFactors,
    VariableDomain,
    create_edges,
    create_variables,
    edge_attr,
    fn_action_masks,
    fn_objects_with_type,
    object_list,
    translate_edges,
)
from ..obs import GraphTypes
from regawa.data.stacked import StackedStringFactorGraph
from regawa.model import (
    BaseModel,
    Grounding,
    GroundObs,
    arity,
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
