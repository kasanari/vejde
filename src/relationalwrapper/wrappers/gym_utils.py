from collections.abc import Callable
from gymnasium.spaces import Discrete, Sequence, Dict, Space, MultiDiscrete


def obs_space(
    num_relations: int, num_types: int, max_arity: int, var_value_space: Space
) -> Dict:
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
            "senders": Sequence(Discrete(2000), stack=True),
            "receivers": Sequence(Discrete(2000), stack=True),
            "edge_attr": Sequence(Discrete(max_arity + 1), stack=True),
            "length": Sequence(Discrete(2000), stack=True),
            "n_nodes": Discrete(2000),
            "global_vars": Sequence(Discrete(2000), stack=True),
            "global_vals": Sequence(var_value_space, stack=True),
            "global_length": Sequence(Discrete(2000), stack=True),
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
