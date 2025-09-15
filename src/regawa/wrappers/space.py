from typing import Any, TypeVar
from gymnasium import Space
from gymnasium.spaces import Box, Discrete, Sequence
import numpy as np
from ..data.data import HeteroObsData, ObsData
from gymnasium.vector.utils.space_utils import batch_differing_spaces

V = TypeVar("V", np.float32, np.bool_)


class FactorGraphSpace(Space[ObsData[V]]):
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
        self.action_arity_mask = Sequence(Box(0, 1, (num_actions,)), stack=True)
        self.action_type_mask = Sequence(Box(0, 1, (num_actions,)), stack=True)
        self.senders = Sequence(Discrete(big_number), stack=True)
        self.receivers = Sequence(Discrete(big_number), stack=True)
        self.edge_attr = Sequence(Discrete(max_arity + 1), stack=True)
        self.length = Sequence(Discrete(big_number), stack=True)
        self.global_vars = Sequence(Discrete(big_number), stack=True)
        self.global_vals = Sequence(var_value_space, stack=True)
        self.global_length = Sequence(Discrete(big_number), stack=True)
        super().__init__(None, None, seed)

    def __eq__(self, other: Any) -> bool:
        """Check whether `other` is equivalent to this instance. Doesn't check dtype equivalence."""
        if not isinstance(other, FactorGraphSpace):
            return False
        
        results = [
            self.var_type == other.var_type,
            self.var_value == other.var_value,
            self.factor == other.factor,
            self.action_arity_mask == other.action_arity_mask,
            self.action_type_mask == other.action_type_mask,
            self.senders == other.senders,
            self.receivers == other.receivers,
            self.edge_attr == other.edge_attr,
            self.length == other.length,
            self.global_vars == other.global_vars,
            self.global_vals == other.global_vals,
            self.global_length == other.global_length,
        ]
        return all(results)

    def __contains__(self, item: ObsData[V]) -> bool:
        """Check whether `item` is in this space."""
        if not isinstance(item, ObsData):  # type: ignore
            return False

        results = [
            item.var_type in self.var_type,
            item.var_value in self.var_value,
            item.factor in self.factor,
            item.action_arity_mask in self.action_arity_mask,
            item.action_type_mask in self.action_type_mask,
            item.v_to_f in self.senders,
            item.f_to_v in self.receivers,
            item.edge_attr in self.edge_attr,
            item.length in self.length,
            item.global_vars in self.global_vars,
            item.global_vals in self.global_vals,
            item.global_length in self.global_length,
        ]

        return all(results)


class HeteroStateSpace(Space[HeteroObsData]):
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
        self.bool = FactorGraphSpace[np.bool_](
            num_relations, num_types, max_arity, num_actions, bool_space
        )
        self.float = FactorGraphSpace[np.float32](
            num_relations, num_types, max_arity, num_actions, number_space
        )
        super().__init__(None, None, seed)

    def __eq__(self, other: Any) -> bool:
        """Check whether `other` is equivalent to this instance. Doesn't check dtype equivalence."""
        if not isinstance(other, HeteroStateSpace):
            return False
        
        results = [
            self.bool.var_type == other.bool.var_type,
            self.bool.var_value == other.bool.var_value,
            self.bool.factor == other.bool.factor,
            self.float.var_type == other.float.var_type,
            self.float.var_value == other.float.var_value,
            self.float.factor == other.float.factor,
        ]

        return all(results)

    def __contains__(self, item: HeteroObsData) -> bool:
        """Check whether `item` is in this space."""
        if not isinstance(item, HeteroObsData):  # type: ignore
            return False

        return item.bool in self.bool and item.float in self.float


@batch_differing_spaces.register(HeteroStateSpace)  # type: ignore
def batch_differing_spaces(spaces: list[HeteroStateSpace]):
    return spaces
