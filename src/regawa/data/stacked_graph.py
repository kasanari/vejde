from regawa.data.actions import ActionMask
from regawa.data.graph import (
    Edges,
    StackedStringVariables,
    StringFactors,
    VariableDomain,
)


from typing import Generic, NamedTuple


class StackedStringFactorGraph(NamedTuple, Generic[VariableDomain]):
    variables: StackedStringVariables[VariableDomain]
    factors: StringFactors
    edges: Edges
    global_variables: StackedStringVariables[VariableDomain]
    action_masks: ActionMask
