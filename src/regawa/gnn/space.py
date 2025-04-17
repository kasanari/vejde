from typing import Any, NamedTuple
import typing
from gymnasium import Space
from gymnasium.spaces import Box, Discrete, Sequence
import numpy as np
from .data import HeteroObs, ObsData
from gymnasium.vector.utils.space_utils import batch_differing_spaces


class FactorGraphSpace(Space[ObsData]):
    def __init__(
        self,
        num_relations: int,
        num_types: int,
        max_arity: int,
        num_actions: int,
        var_value_space: Space[Any],
        seed: int | np.random.Generator | None = None,
    ):
        big_number = 2000

        self.var_type = Sequence(Discrete(num_relations), stack=True)
        self.var_value = Sequence(var_value_space, stack=True)
        self.factor = Sequence(
            Discrete(
                num_types,
            ),
            stack=True,
        )
        self.action_mask = Sequence(Box(0, 1, (num_actions,)), stack=True)
        self.senders = Sequence(Discrete(big_number), stack=True)
        self.receivers = Sequence(Discrete(big_number), stack=True)
        self.edge_attr = Sequence(Discrete(max_arity + 1), stack=True)
        self.length = Sequence(Discrete(big_number), stack=True)
        self.global_vars = Sequence(Discrete(big_number), stack=True)
        self.global_vals = Sequence(var_value_space, stack=True)
        self.global_length = Sequence(Discrete(big_number), stack=True)
        super().__init__(None, None, seed)


class HeteroStateSpace(Space[HeteroObs]):
    def __init__(
        self,
        num_types: int,
        num_relations: int,
        max_arity: int,
        num_actions: int,
        seed: int | np.random.Generator | None = None,
    ):
        bool_space = Discrete(2)
        number_space = Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(),
        )
        self.bool = FactorGraphSpace(
            num_relations, num_types, max_arity, num_actions, bool_space
        )
        self.float = FactorGraphSpace(
            num_relations, num_types, max_arity, num_actions, number_space
        )
        super().__init__(None, None, seed)

    def __eq__(self, other: Any) -> bool:
        """Check whether `other` is equivalent to this instance. Doesn't check dtype equivalence."""
        if not isinstance(other, HeteroStateSpace):
            return False

        return (
            self.bool.var_type == other.bool.var_type
            and self.bool.var_value == other.bool.var_value
            and self.bool.factor == other.bool.factor
            and self.float.var_type == other.float.var_type
            and self.float.var_value == other.float.var_value
            and self.float.factor == other.float.factor
        )


@batch_differing_spaces.register(HeteroStateSpace)
def batch_differing_spaces(spaces: list[HeteroStateSpace]):
    return spaces
