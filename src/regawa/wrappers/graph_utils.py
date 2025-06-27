from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar

import numpy as np

from regawa import BaseModel
from regawa.data.data import HeteroObsData
from regawa.model import GroundValue
from regawa.model.utils import (
    fn_valid_action_fluents_given_arity,
    fn_valid_action_fluents_given_type,
)
from regawa.wrappers.grounding_utils import (
    bool_groundings,
    fn_is_bool,
    fn_is_numeric,
    numeric_groundings,
    fn_objects_with_type,
)
from regawa.wrappers.gym_utils import idxgraph_to_obsdata
from regawa.wrappers.types import FactorGraph, HeteroGraph, Variables
from regawa.wrappers.utils import (
    generate_bipartite_obs_func,
    map_graph_to_idx,
    object_list,
)

V = TypeVar("V")


def _map_graph_to_idx(
    factorgraph: FactorGraph[V],
    rel_to_idx: Callable[[str], int],
    type_to_idx: Callable[[str], int],
    var_val_dtype: type,
):
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


def fn_heterograph_to_heteroobs(
    model: BaseModel,
):
    def heterograph_to_heteroobs(heterogenous_graph: HeteroGraph) -> HeteroObsData:
        return HeteroObsData(
            bool=idxgraph_to_obsdata(
                _map_graph_to_idx(
                    heterogenous_graph.boolean,  # type: ignore
                    model.fluent_to_idx,
                    model.type_to_idx,
                    np.int8,
                ),
            ),
            float=idxgraph_to_obsdata(
                _map_graph_to_idx(
                    heterogenous_graph.numeric,  # type: ignore
                    model.fluent_to_idx,
                    model.type_to_idx,
                    np.float32,
                ),
            ),
        )

    return heterograph_to_heteroobs


def fn_obsdict_to_graph(
    model: BaseModel,
):
    objects_with_type = fn_objects_with_type(model.fluent_param)
    is_numeric, is_bool = (
        fn_is_numeric(model.fluent_range),
        fn_is_bool(model.fluent_range),
    )
    valid_action_type_func, valid_action_arity_func = (
        fn_valid_action_fluents_given_type(model),
        fn_valid_action_fluents_given_arity(model),
    )
    generate_bipartite_obs_bool, generate_bipartite_obs_numeric = (
        generate_bipartite_obs_func(
            FactorGraph[bool],
            valid_action_type_func,
            valid_action_arity_func,
        ),
        generate_bipartite_obs_func(
            FactorGraph[float],
            valid_action_type_func,
            valid_action_arity_func,
        ),
    )
    b_g, n_g = (
        partial(
            bool_groundings,
            is_bool=is_bool,
        ),
        partial(
            numeric_groundings,
            is_numeric=is_numeric,
        ),
    )

    def obsdict_to_graph(rddl_obs: dict[GroundValue, Any]):
        filtered_groundings = [
            g
            for g in rddl_obs
            if rddl_obs[g] is not None  # type: ignore
        ]

        filtered_obs: dict[GroundValue, Any] = {
            k: rddl_obs[k] for k in filtered_groundings
        }

        object_nodes = object_list(list(filtered_obs.keys()), objects_with_type)

        bool_g, numeric_g = (
            generate_bipartite_obs_bool(
                filtered_obs,
                b_g(filtered_groundings),
                object_nodes,
            ),
            generate_bipartite_obs_numeric(
                filtered_obs,
                n_g(filtered_groundings),
                object_nodes,
            ),
        )

        assert isinstance(bool_g, FactorGraph)
        assert isinstance(numeric_g, FactorGraph)
        return HeteroGraph(numeric_g, bool_g), filtered_groundings

    return obsdict_to_graph
