from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

from regawa import BaseModel
from regawa.gnn.data import HeteroObs
from regawa.model import GroundValue
from regawa.model.utils import valid_action_fluents
from regawa.wrappers.grounding_utils import (
    bool_groundings,
    is_bool_func,
    is_numeric_func,
    numeric_groundings,
    objects_with_type_func,
)
from regawa.wrappers.gym_utils import graph_to_dict
from regawa.wrappers.util_types import FactorGraph, HeteroGraph, Variables
from regawa.wrappers.utils import generate_bipartite_obs, map_graph_to_idx, object_list

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
        factorgraph.action_mask,
        factorgraph.factor_types,
        rel_to_idx,
        type_to_idx,
        var_val_dtype,
    )


def create_obs_dict_func(
    model: BaseModel,
):
    def f(heterogenous_graph: HeteroGraph) -> dict[str, Any]:
        return HeteroObs(
            bool=graph_to_dict(
                _map_graph_to_idx(
                    heterogenous_graph.boolean,  # type: ignore
                    model.fluent_to_idx,
                    model.type_to_idx,
                    np.int8,
                ),
            ),
            float=graph_to_dict(
                _map_graph_to_idx(
                    heterogenous_graph.numeric,  # type: ignore
                    model.fluent_to_idx,
                    model.type_to_idx,
                    np.float32,
                ),
            ),
        )

    return f


def create_graphs_func(
    model: BaseModel,
):
    is_numeric = is_numeric_func(model.fluent_range)
    is_bool = is_bool_func(model.fluent_range)
    valid_actions = valid_action_fluents(model)
    objects_with_type = objects_with_type_func(model.fluent_param)

    def f(rddl_obs: dict[GroundValue, Any]):
        filtered_groundings = [
            g
            for g in rddl_obs
            if rddl_obs[g] is not None  # type: ignore
        ]

        filtered_obs: dict[GroundValue, Any] = {
            k: rddl_obs[k] for k in filtered_groundings
        }

        bool_ground = bool_groundings(filtered_groundings, is_bool)

        object_nodes = object_list(list(filtered_obs.keys()), objects_with_type)

        bool_g = generate_bipartite_obs(
            FactorGraph[bool],
            filtered_obs,
            bool_ground,
            objects_with_type,
            valid_actions,
            model.num_actions,
            object_nodes,
        )

        n_g = numeric_groundings(filtered_groundings, is_numeric)

        numeric_g = generate_bipartite_obs(
            FactorGraph[float],
            filtered_obs,
            n_g,
            objects_with_type,
            valid_actions,
            model.num_actions,
            object_nodes,
        )

        assert isinstance(bool_g, FactorGraph)
        assert isinstance(numeric_g, FactorGraph)
        # hetero_g = HeteroGraph(numeric_g, bool_g)
        return HeteroGraph(numeric_g, bool_g), filtered_groundings

    return f
