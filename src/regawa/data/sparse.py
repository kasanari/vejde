from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


from typing import Generic, NamedTuple, TypeVar

V = TypeVar("V", np.float32, np.bool_, np.int64, np.int8)


class SparseArray(NamedTuple, Generic[V]):
    """
    This is a simple sparse COOrdinate array representation.
    index is the position of the values in the original dense array, e.g. the graph the node belongs to

    assuming [1, 2, 3], [4, 5] and [6, 7, 8, 9], the sparse representation will be
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    indices = [0, 0, 0, 1, 1, 2, 2, 2, 2]
    """

    values: NDArray[V]
    indices: NDArray[np.int64]

    @property
    def shape(self):
        return self.values.shape

    def concat(self, other: SparseArray[V]) -> SparseArray[V]:
        return SparseArray(
            np.concatenate((self.values, other.values)),
            np.concatenate((self.indices, other.indices)),
        )
