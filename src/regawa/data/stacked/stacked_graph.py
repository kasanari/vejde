from collections.abc import Sequence
from typing import NamedTuple

from regawa.data.graph import (
    StringFactors,
    VariableDomain,
)
from regawa.data.graph.graph import ActionMask, Edges
from regawa.model import Grounding


class StackedStringVariables[T: VariableDomain](NamedTuple):
    types: Sequence[str]
    values: Sequence[Sequence[T]]
    length: Sequence[int]
    n_variable: int
    groundings: Sequence[Grounding]


class StackedStringFactorGraph[T: VariableDomain](NamedTuple):
    variables: StackedStringVariables[T]
    factors: StringFactors
    edges: Edges
    global_variables: StackedStringVariables[T]
    action_masks: ActionMask
