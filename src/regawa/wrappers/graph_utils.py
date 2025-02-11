from regawa import BaseModel
from regawa.model import GroundValue
from regawa.model.utils import valid_action_fluents
from regawa.wrappers.grounding_utils import bool_groundings, numeric_groundings
from typing import Any
from regawa.wrappers.gym_utils import graph_to_dict
from regawa.wrappers.util_types import (
    FactorGraph,
    HeteroGraph,
    IdxFactorGraph,
    Variables,
)


import numpy as np


from collections.abc import Callable

from regawa.wrappers.utils import generate_bipartite_obs


def map_graph_to_idx[V](
    factorgraph: FactorGraph[V],
    rel_to_idx: Callable[[str], int],
    type_to_idx: Callable[[str], int],
    var_val_dtype: type,
) -> IdxFactorGraph[V]:
    edge_attributes = np.asarray(factorgraph.edge_attributes, dtype=np.int64)

    vals = np.array(  # type: ignore
        factorgraph.variable_values, dtype=var_val_dtype
    )  # TODO: handle None values in a better way

    global_vars_values = np.array(  # type: ignore
        factorgraph.global_variable_values, dtype=var_val_dtype
    )
    global_vars = np.array(
        [rel_to_idx(key) for key in factorgraph.global_variables], dtype=np.int64
    )

    return IdxFactorGraph(
        Variables(
            np.array(
                [rel_to_idx(key) for key in factorgraph.variables], dtype=np.int64
            ),
            vals,
            lengths=np.ones_like(vals, dtype=np.int64),
        ),
        np.array(
            [type_to_idx(object) for object in factorgraph.factor_values],
            dtype=np.int64,
        ),
        factorgraph.senders,
        factorgraph.receivers,
        edge_attributes,
        Variables(
            global_vars,
            global_vars_values,
            lengths=np.ones_like(global_vars_values, dtype=np.int64),
        ),
        action_mask=factorgraph.action_mask,
    )


def create_obs_dict(
    heterogenous_graph: HeteroGraph,
    model: BaseModel,
) -> dict[str, Any]:
    obs_boolean = graph_to_dict(
        map_graph_to_idx(
            heterogenous_graph.boolean,  # type: ignore
            model.fluent_to_idx,
            model.type_to_idx,
            np.int8,
        ),
    )

    obs_numeric = graph_to_dict(
        map_graph_to_idx(  # type: ignore
            heterogenous_graph.numeric,  # type: ignore
            model.fluent_to_idx,
            model.type_to_idx,
            np.float32,
        )
    )

    obs = {
        "bool": obs_boolean,
        "float": obs_numeric,
    }

    return obs


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
