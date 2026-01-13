from typing import Any
from gymnasium import Space
from gymnasium.spaces import Box, Discrete, Sequence
import numpy as np

from regawa.data.graph import VariableDomain

from .obs import HeteroObsData
from .obs import ObsData
from .graph import Variables


BIG_NUMBER = 2000


class VariableSpace(Space[Variables[VariableDomain]]):
    def __init__(self, num_relations: int, var_value_space: Space[VariableDomain]):
        self.var_type = Sequence(Discrete(num_relations), stack=True)
        self.var_value = Sequence(var_value_space, stack=True)
        self.length = Sequence(Discrete(BIG_NUMBER), stack=True)

    def __eq__(self, other: Any) -> bool:
        """Check whether `other` is equivalent to this instance. Doesn't check dtype equivalence."""
        if not isinstance(other, VariableSpace):
            return False

        results = [
            self.var_type == other.var_type,
            self.var_value == other.var_value,
            self.length == other.length,
        ]
        return all(results)

    def __contains__(self, item: Any) -> bool:
        """Check whether `item` is in this space."""
        if isinstance(item, Variables):
            return False

        results = [
            item.types in self.var_type,
            item.value in self.var_value,
            item.length in self.length,
        ]

        return all(results)


class FactorGraphSpace(Space[ObsData[VariableDomain]]):
    def __init__(
        self,
        num_relations: int,
        num_types: int,
        max_arity: int,
        num_actions: int,
        var_value_space: Space[Any],
        seed: int | np.random.Generator | None = None,
    ):
        self.var = VariableSpace(num_relations, var_value_space)

        self.factor = Sequence(
            Discrete(
                num_types,
            ),
            stack=True,
        )
        self.action_arity_mask = Sequence(Box(0, 1, (num_actions,)), stack=True)
        self.action_type_mask = Sequence(Box(0, 1, (num_actions,)), stack=True)
        self.senders = Sequence(Discrete(BIG_NUMBER), stack=True)
        self.receivers = Sequence(Discrete(BIG_NUMBER), stack=True)
        self.edge_attr = Sequence(Discrete(max_arity + 1), stack=True)

        self.global_var = VariableSpace(num_relations, var_value_space)
        super().__init__(None, None, seed)

    def __eq__(self, other: Any) -> bool:
        """Check whether `other` is equivalent to this instance. Doesn't check dtype equivalence."""
        if not isinstance(other, FactorGraphSpace):
            return False

        results = [
            self.var == other.var,
            self.factor == other.factor,
            self.action_arity_mask == other.action_arity_mask,
            self.action_type_mask == other.action_type_mask,
            self.senders == other.senders,
            self.receivers == other.receivers,
            self.edge_attr == other.edge_attr,
            self.global_var == other.global_var,
        ]
        return all(results)

    def __contains__(self, item: ObsData[VariableDomain]) -> bool:
        """Check whether `item` is in this space."""
        if not isinstance(item, ObsData):  # type: ignore
            return False

        results = [
            item.var in self.var,
            item.factor.types in self.factor,
            item.action_masks.action_arity_mask in self.action_arity_mask,
            item.action_masks.action_type_mask in self.action_type_mask,
            item.edges.v_to_f in self.senders,
            item.edges.f_to_v in self.receivers,
            item.edges.edge_attr in self.edge_attr,
            item.global_var in self.global_var,
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
        bool_space: Discrete[np.int8] = Discrete(2)
        number_space = Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(),
        )
        self.bool = FactorGraphSpace[np.int8](
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
            self.bool.var == other.bool.var,
            self.bool.factor == other.bool.factor,
            self.float.var == other.float.var,
            self.float.factor == other.float.factor,
        ]

        return all(results)

    def __contains__(self, item: HeteroObsData) -> bool:
        """Check whether `item` is in this space."""
        if not isinstance(item, HeteroObsData):  # type: ignore
            return False

        return item.bool in self.bool and item.float in self.float
