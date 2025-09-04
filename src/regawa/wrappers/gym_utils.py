from collections.abc import Callable
from typing import Any, TypeVar

from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Sequence, Space
import numpy as np
from regawa.gnn.data import ObsData
from regawa.wrappers.space import HeteroStateSpace
from regawa.wrappers.util_types import IdxFactorGraph

V = TypeVar("V", np.float32, np.bool_)


def max_arity(observation_space: Dict) -> int:
    return observation_space.bool.edge_attr.feature_space.n - 1  # type: ignore


def obs_space(
    num_relations: int,
    num_types: int,
    max_arity: int,
    num_actions: int,
    var_value_space: Space[Any],
) -> Dict:
    big_number = 2000
    return Dict(
        {  # type: ignore
            "var_type": Sequence(Discrete(num_relations), stack=True),
            "var_value": Sequence(var_value_space, stack=True),
            "factor": Sequence(
                Discrete(
                    num_types,
                ),
                stack=True,
            ),
            "action_mask": Sequence(Box(0, 1, (num_actions,)), stack=True),
            "senders": Sequence(Discrete(big_number), stack=True),
            "receivers": Sequence(Discrete(big_number), stack=True),
            "edge_attr": Sequence(Discrete(max_arity + 1), stack=True),
            "length": Sequence(Discrete(big_number), stack=True),
            "global_vars": Sequence(Discrete(big_number), stack=True),
            "global_vals": Sequence(var_value_space, stack=True),
            "global_length": Sequence(Discrete(big_number), stack=True),
        }
    )


def action_space(
    action_fluents: list[str],
    num_actions: int,
    num_objects: int,
    arity: Callable[[str], int],
) -> MultiDiscrete:
    max_action_args = max(arity(a) for a in action_fluents) or 1

    return MultiDiscrete(
        [num_actions]
        + [
            num_objects,
        ]
        * max_action_args
    )


def n_types(observation_space: HeteroStateSpace):
    return int(observation_space.bool.factor.feature_space.n)  # type: ignore


def n_relations(observation_space: HeteroStateSpace):
    return int(observation_space.bool.var_type.feature_space.n)  # type: ignore


def n_actions(action_space: MultiDiscrete):
    return int(action_space.nvec[0])  # type: ignore


def idxgraph_to_obsdata(idx_g: IdxFactorGraph[V]):
    return ObsData[V](
        var_type=idx_g.variables.types,
        var_value=idx_g.variables.values,
        factor=idx_g.factors,
        v_to_f=idx_g.senders,
        f_to_v=idx_g.receivers,
        edge_attr=idx_g.edge_attributes,
        length=idx_g.variables.lengths,
        global_vars=idx_g.global_vars.types,
        global_vals=idx_g.global_vars.values,
        global_length=idx_g.global_vars.lengths,
        action_arity_mask=idx_g.action_arity_mask,
        action_type_mask=idx_g.action_type_mask,
        n_factor=idx_g.factors.shape[0],  # + obs["var_value"].shape[0]
        n_variable=idx_g.variables.lengths.shape[0],
    )
