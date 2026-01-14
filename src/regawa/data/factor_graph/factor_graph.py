from typing import Generic, NamedTuple

from regawa.data.graph import StringFactors, StringVariables, VariableDomain
from regawa.data.graph.graph import ActionMask, Edges


class StringFactorGraph(NamedTuple, Generic[VariableDomain]):
    """A FactorGraph with string attributes."""

    variables: StringVariables[VariableDomain]
    factors: StringFactors
    edges: Edges
    global_variables: StringVariables[VariableDomain]
    action_masks: ActionMask
