from collections.abc import Callable
from typing import Any
from gymnasium.spaces import Discrete, Sequence, Dict, Space, MultiDiscrete, Box


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
    max_action_args = max(arity(a) for a in action_fluents)

    return MultiDiscrete(
        [num_actions]
        + [
            num_objects,
        ]
        * max_action_args
    )


def n_types(observation_space: Dict):
    return int(observation_space["bool"]["factor"].feature_space.n)  # type: ignore


def n_relations(observation_space: Dict):
    return int(observation_space["bool"]["var_type"].feature_space.n)  # type: ignore


def n_actions(action_space: MultiDiscrete):
    return int(action_space.nvec[0])  # type: ignore
