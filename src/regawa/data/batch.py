import numpy as np
from numpy.typing import NDArray

from typing import Generic, NamedTuple, TypeVar


from .actions import ActionMask

from .graph import VariableDomain, Edges

from .sparse import SparseArray


class BatchedVariables(NamedTuple, Generic[VariableDomain]):
    var_value: SparseArray[VariableDomain]
    var_type: SparseArray[np.int64]
    n_variable: NDArray[np.int64]
    length: NDArray[np.int64]
    times: NDArray[np.int64]


class BatchedFactors(NamedTuple):
    factor: SparseArray[np.int64]
    n_factor: NDArray[np.int64]


class BatchData(NamedTuple, Generic[VariableDomain]):
    """This represents a batch of multiple factor graphs."""

    factor: BatchedFactors
    variables: BatchedVariables[VariableDomain]
    edges: Edges
    n_graphs: np.int64
    global_variables: BatchedVariables[VariableDomain]
    action_masks: ActionMask


class HeteroBatchData(NamedTuple):
    """This represents a batch of multiple heterogeneous factor graphs."""

    boolean: BatchData[np.int8]
    numeric: BatchData[np.float32]

    @property
    def n_graphs(self) -> np.int64:
        return self.boolean.n_graphs

    @property
    def n_factor(self) -> NDArray[np.int64]:
        # The assumption at the moment is that both boolean and numeric use the same factors, even if one might have no variables.
        return self.boolean.factor.n_factor


ArrayDomain = TypeVar("ArrayDomain", np.int8, np.float32, np.bool_, np.int64)
