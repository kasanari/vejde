from regawa import BaseModel
from typing import Any
from regawa.model import GroundValue
from regawa.model.utils import valid_action_fluents
from regawa.wrappers.grounding_utils import bool_groundings, numeric_groundings
from regawa.wrappers.gym_utils import graph_to_dict
from regawa.wrappers.util_types import (
    HeteroGraph,
    IdxFactorGraph,
    StackedFactorGraph,
    Variables,
)


import numpy as np


from collections.abc import Callable
from itertools import chain

from regawa.wrappers.utils import (
    generate_bipartite_obs,
)


def map_stacked_graph_to_idx[V](
    factorgraph: StackedFactorGraph[V],
    rel_to_idx: Callable[[str], int],
    type_to_idx: Callable[[str], int],
    val_dtype: type,
) -> IdxFactorGraph[V]:
    edge_attributes = np.asarray(factorgraph.edge_attributes, dtype=np.int64)

    # Flatten the list of node history lists to account for different node history lengths
    vals = list(chain(*factorgraph.variable_values))

    vars = [
        [factorgraph.variables[i] for _ in v]
        for i, v in enumerate(factorgraph.variable_values)
    ]
    vars = list(chain(*vars))

    vars = [rel_to_idx(p) for p in vars]

    factors = [type_to_idx(object) for object in factorgraph.factor_values]

    global_vars_values = list(chain(*factorgraph.global_variable_values))

    global_vars = [
        [factorgraph.global_variables[i] for _ in v]
        for i, v in enumerate(factorgraph.global_variable_values)
    ]

    global_vars = list(chain(*global_vars))
    global_vars = [rel_to_idx(p) for p in global_vars]

    global_lengths = np.array(
        [len(v) for v in factorgraph.global_variable_values], dtype=np.int64
    )

    lengths = np.array([len(v) for v in factorgraph.variable_values], dtype=np.int64)

    return IdxFactorGraph(
        Variables(
            np.array(vars, dtype=np.int64),
            np.array(vals, dtype=val_dtype),
            lengths,
        ),
        np.array(
            factors,
            dtype=np.int64,
        ),
        factorgraph.senders,
        factorgraph.receivers,
        edge_attributes,
        Variables(
            np.array(
                global_vars,
            ),
            np.array(global_vars_values, dtype=val_dtype),
            global_lengths,
        ),
        action_mask=factorgraph.action_mask,
    )


def create_stacked_obs(
    heterograph: HeteroGraph,
    model: BaseModel,
) -> dict[str, Any]:
    obs_boolean = graph_to_dict(
        map_stacked_graph_to_idx(  # type: ignore
            heterograph.boolean,  # type: ignore
            model.fluent_to_idx,
            model.type_to_idx,
            np.int64,
        )
    )

    obs_numeric = graph_to_dict(
        map_stacked_graph_to_idx(  # type: ignore
            heterograph.numeric,  # type: ignore
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


def create_stacked_graphs(
    rddl_obs: dict[GroundValue, Any],
    model: BaseModel,
):
    filtered_groundings = sorted([g for g in rddl_obs])

    filtered_obs = {k: rddl_obs[k] for k in filtered_groundings}

    bool_ground = bool_groundings(filtered_groundings, model.fluent_range)

    bool_g = generate_bipartite_obs(
        StackedFactorGraph[bool],
        filtered_obs,
        bool_ground,
        model.fluent_param,
        valid_action_fluents(model),
        model.num_actions,
    )

    n_g = numeric_groundings(filtered_groundings, model.fluent_range)

    numeric_g = generate_bipartite_obs(
        StackedFactorGraph[float],
        filtered_obs,
        n_g,
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
