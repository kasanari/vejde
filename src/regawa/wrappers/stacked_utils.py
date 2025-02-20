from collections.abc import Callable
from itertools import chain
from typing import Any, TypeVar

import numpy as np

from regawa import BaseModel
from regawa.model import GroundValue
from regawa.model.utils import valid_action_fluents
from regawa.wrappers.grounding_utils import bool_groundings, numeric_groundings
from regawa.wrappers.gym_utils import graph_to_dict
from regawa.wrappers.util_types import HeteroGraph, StackedFactorGraph, Variables
from regawa.wrappers.utils import generate_bipartite_obs, map_graph_to_idx

V = TypeVar("V")


def flatten(vals: list[list[V]], vars: list[str]) -> Variables[V]:
    # Flatten the list of node history lists to account for different node history lengths
    flat_vals = list(chain(*vals))
    v = [[vars[i] for _ in v] for i, v in enumerate(vals)]  # expand the variable names
    flat_vars = list(chain(*v))
    lengths = [len(v) for v in vals]
    return Variables(flat_vars, flat_vals, lengths)


def flatten_values(
    factorgraph: StackedFactorGraph[V],
) -> tuple[Variables[V], Variables[V]]:
    return (
        flatten(factorgraph.variable_values, factorgraph.variables),
        flatten(factorgraph.global_variable_values, factorgraph.global_variables),
    )


def _map_graph_to_idx(
    factorgraph: StackedFactorGraph[V],
    rel_to_idx: Callable[[str], int],
    type_to_idx: Callable[[str], int],
    var_val_dtype: type,
):
    return map_graph_to_idx(
        *flatten_values(factorgraph),
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

    bool_g = generate_bipartite_obs(
        StackedFactorGraph[bool],
        filtered_obs,
        bool_groundings(filtered_groundings, model.fluent_range),
        model.fluent_param,
        valid_action_fluents(model),
        model.num_actions,
    )

    numeric_g = generate_bipartite_obs(
        StackedFactorGraph[float],
        filtered_obs,
        numeric_groundings(filtered_groundings, model.fluent_range),
        model.fluent_param,
        valid_action_fluents(model),
        model.num_actions,
    )

    assert isinstance(
        bool_g, StackedFactorGraph
    ), f"expected StackedFactorGraph but got {type(bool_g)}"
    assert isinstance(
        numeric_g, StackedFactorGraph
    ), f"expected StackedFactorGraph but got {type(numeric_g)}"
    return HeteroGraph(numeric_g, bool_g), filtered_groundings
