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


class IndexedFactorGraph(NamedTuple, Generic[VariableDomain]):
    """
    This class represents an factor graph of groundings and objects with numeric indexes as identifiers.
    Assume a grounding p(o) = v.
    """

    var: Variables[VariableDomain]  # grounding variables
    factor: Factors  # grounding factors/objects
    edges: Edges  # edges between groundings and objects
    global_var: Variables[VariableDomain]  # global grounding variables
    action_masks: ActionMask  # action masks


class HeteroIndexedFactorGraph(NamedTuple):
    """
    This class represents a heterogeneous factor graph with boolean and float features.
    """

    bool: IndexedFactorGraph[np.int8]  # boolean Obs
    float: IndexedFactorGraph[np.float32]  # numeric Obs
