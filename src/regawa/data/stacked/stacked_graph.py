from collections.abc import Sequence
from typing import Generic, NamedTuple

from regawa.data.graph import (
    StringFactors,
    VariableDomain,
)
from regawa.data.graph.graph import ActionMask, Edges
from regawa.model import Grounding


class StackedStringVariables(NamedTuple, Generic[VariableDomain]):
    types: Sequence[str]
    values: Sequence[Sequence[VariableDomain]]
    length: Sequence[int]
    n_variable: int
    groundings: Sequence[Grounding]


class StackedStringFactorGraph(NamedTuple, Generic[VariableDomain]):
    variables: StackedStringVariables[VariableDomain]
    factors: StringFactors
    edges: Edges
    global_variables: StackedStringVariables[VariableDomain]
    action_masks: ActionMask
