from typing import Generic, NamedTuple, TypeVar

import numpy as np
from numpy.typing import NDArray

from regawa.data.factor_graph import StringFactorGraph
from regawa.data.graph import ActionMask, VariableDomain, Variables
from regawa.data.graph.graph import Edges
from regawa.data.stacked import StackedStringFactorGraph

GraphTypes = TypeVar(
    "GraphTypes",
    StringFactorGraph[np.int8],
    StackedStringFactorGraph[np.int8],
    StringFactorGraph[np.float32],
    StackedStringFactorGraph[np.float32],
)


class Factors(NamedTuple):
    types: NDArray[np.int64]  # object of grounding, e.g. "o"
    n_factor: int  # number of objects/factors.


class ObsData(NamedTuple, Generic[VariableDomain]):
    """
    This class represents a factor graph of groundings and objects.
    Assume a grounding p(o) = v.
    """

    var: Variables[VariableDomain]  # grounding variables
    factor: Factors  # grounding factors/objects
    edges: Edges  # edges between groundings and objects
    global_var: Variables[VariableDomain]  # global grounding variables
    action_masks: ActionMask  # action masks


class HeteroObsData(NamedTuple):
    """
    This class represents a heterogeneous observation with boolean and float features.
    """

    bool: ObsData[np.int8]  # boolean ObsData
    float: ObsData[np.float32]  # numeric ObsData
