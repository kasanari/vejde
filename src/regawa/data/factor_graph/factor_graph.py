from typing import NamedTuple

from regawa.data.graph import StringFactors, StringVariables, VariableDomain
from regawa.data.graph.graph import ActionMask, Edges


class StringFactorGraph[T: VariableDomain](NamedTuple):
    """A FactorGraph with string attributes."""

    variables: StringVariables[T]
    factors: StringFactors
    edges: Edges
    global_variables: StringVariables[T]
    action_masks: ActionMask
