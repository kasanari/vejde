from regawa import BaseModel
from regawa.model import GroundValue
from regawa.model.utils import valid_action_fluents
from regawa.wrappers.grounding_utils import bool_groundings, numeric_groundings
from typing import Any
from regawa.wrappers.gym_utils import graph_to_dict
from regawa.wrappers.util_types import (
    FactorGraph,
    HeteroGraph,
    Variables,
)

import numpy as np


from collections.abc import Callable

from regawa.wrappers.utils import generate_bipartite_obs, map_graph_to_idx


def _map_graph_to_idx[V](
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


def create_obs_dict(
    heterogenous_graph: HeteroGraph,
    model: BaseModel,
) -> dict[str, Any]:
    return {
        k: graph_to_dict(
            _map_graph_to_idx(
                v,  # type: ignore
                model.fluent_to_idx,
                model.type_to_idx,
                dtype,
            ),
        )
        for k, v, dtype in [
            ("bool", heterogenous_graph.boolean, np.int8),
            ("float", heterogenous_graph.numeric, np.float32),
        ]
    }


def create_graphs(
    rddl_obs: dict[GroundValue, Any],
    model: BaseModel,
):
    filtered_groundings = sorted(
        [
            g
            for g in rddl_obs
            if rddl_obs[g] is not None  # type: ignore
        ]
    )

    filtered_obs: dict[GroundValue, Any] = {k: rddl_obs[k] for k in filtered_groundings}

    bool_ground = bool_groundings(filtered_groundings, model.fluent_range)

    bool_g = generate_bipartite_obs(
        FactorGraph[bool],
        filtered_obs,
        bool_ground,
        model.fluent_param,
        valid_action_fluents(model),
        model.num_actions,
    )

    n_g = numeric_groundings(filtered_groundings, model.fluent_range)

    numeric_g = generate_bipartite_obs(
        FactorGraph[float],
        filtered_obs,
        n_g,
        model.fluent_param,
        valid_action_fluents(model),
        model.num_actions,
    )

    assert isinstance(bool_g, FactorGraph)
    assert isinstance(numeric_g, FactorGraph)
    # hetero_g = HeteroGraph(numeric_g, bool_g)
    return HeteroGraph(numeric_g, bool_g), filtered_groundings
