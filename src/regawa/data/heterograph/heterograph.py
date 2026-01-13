from typing import NamedTuple

import numpy as np

from regawa.data.factor_graph import StringFactorGraph
from regawa.data.stacked import StackedStringFactorGraph


class HeteroGraph(NamedTuple):
    numeric: StringFactorGraph[np.float32] | StackedStringFactorGraph[np.float32]
    boolean: StringFactorGraph[np.int8] | StackedStringFactorGraph[np.int8]
