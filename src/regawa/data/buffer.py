from collections import deque
from collections.abc import Iterable
from typing import Generic

import numpy as np

from regawa.data.batch import (
    Batch,
    HeteroBatch,
    create_batch,
)
from regawa.data.graph import (
    VariableDomain,
)
from regawa.data.obs import (
    HeteroIndexedFactorGraph,
    IndexedFactorGraph,
)


class GraphBuffer(Generic[VariableDomain]):
    def __init__(self) -> None:
        self.data: deque[IndexedFactorGraph[VariableDomain]] = deque()

    def extend(self, obs: Iterable[IndexedFactorGraph[VariableDomain]]) -> None:
        self.data.extend(obs)

    def add_single(self, obs: IndexedFactorGraph[VariableDomain]) -> None:
        self.data.append(obs)

    def add_single_dict(self, obs: IndexedFactorGraph[VariableDomain]) -> None:
        self.data.append(obs)

    def batch(self) -> Batch[VariableDomain]:
        return create_batch(list(self.data))

    def __getitem__(self, index: int) -> IndexedFactorGraph[VariableDomain]:
        return self.data[index]

    def minibatch(self, indices: Iterable[int]) -> Batch[VariableDomain]:
        return create_batch([self.data[i] for i in indices])


class HeteroGraphBuffer:
    def __init__(self) -> None:
        self.boolean = GraphBuffer[np.int8]()
        self.numeric = GraphBuffer[np.float32]()

    def extend(self, obs: list[HeteroIndexedFactorGraph]) -> None:
        for o in obs:
            self.boolean.add_single(o.bool)
            self.numeric.add_single(o.float)

    def add_single_dict(self, obs: HeteroIndexedFactorGraph) -> None:
        self.boolean.add_single_dict(obs.bool)
        self.numeric.add_single_dict(obs.float)

    @property
    def batch(self) -> HeteroBatch:
        return HeteroBatch(
            boolean=create_batch(list(self.boolean.data)),
            numeric=create_batch(list(self.numeric.data)),
        )

    def minibatch(self, indices: Iterable[int]) -> HeteroBatch:
        return HeteroBatch(
            boolean=create_batch([self.boolean.data[i] for i in indices]),
            numeric=create_batch([self.numeric.data[i] for i in indices]),
        )
