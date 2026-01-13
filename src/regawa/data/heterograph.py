from regawa.data.graph import StringFactorGraph
from regawa.data.stacked_graph import StackedStringFactorGraph


import numpy as np


from typing import NamedTuple


class HeteroGraph(NamedTuple):
    numeric: StringFactorGraph[np.float32] | StackedStringFactorGraph[np.float32]
    boolean: StringFactorGraph[np.bool_] | StackedStringFactorGraph[np.bool_]
