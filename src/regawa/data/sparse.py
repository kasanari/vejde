from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from .func import VariableDomain


class SparseArray[T: VariableDomain](NamedTuple):
    """
    This is a simple sparse COOrdinate array representation.
    index is the position of the values in the original dense array, e.g. the graph the node belongs to

    assuming [1, 2, 3], [4, 5] and [6, 7, 8, 9], the sparse representation will be
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    indices = [0, 0, 0, 1, 1, 2, 2, 2, 2]
    """

    values: NDArray[T]
    indices: NDArray[np.int64]

    @property
    def shape(self):
        return self.values.shape

    def concat(self, other: SparseArray[T]) -> SparseArray[T]:
        return SparseArray(
            np.concatenate((self.values, other.values)),
            np.concatenate((self.indices, other.indices)),
        )
