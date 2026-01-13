import numpy as np


from typing import Generic, NamedTuple


from .actions import ActionMask
from .graph import Edges, Factors, Variables, VariableDomain


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
