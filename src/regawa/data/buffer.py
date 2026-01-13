import numpy as np
from regawa.data.batch import BatchData, HeteroBatchData
from regawa.data.batch_func import batch
from regawa.data.graph import VariableDomain
from regawa.data.obs import ObsData


from collections import deque
from collections.abc import Iterable
from typing import Generic

from regawa.data.obs import HeteroObsData


class GraphBuffer(Generic[VariableDomain]):
    def __init__(self) -> None:
        self.data: deque[ObsData[VariableDomain]] = deque()

    def extend(self, obs: Iterable[ObsData[VariableDomain]]) -> None:
        self.data.extend(obs)

    def add_single(self, obs: ObsData[VariableDomain]) -> None:
        self.data.append(obs)

    def add_single_dict(self, obs: ObsData[VariableDomain]) -> None:
        self.data.append(obs)

    def batch(self) -> BatchData[VariableDomain]:
        return batch(list(self.data))

    def __getitem__(self, index: int) -> ObsData[VariableDomain]:
        return self.data[index]

    def minibatch(self, indices: Iterable[int]) -> BatchData[VariableDomain]:
        return batch([self.data[i] for i in indices])


class HeteroGraphBuffer:
    def __init__(self) -> None:
        self.boolean = GraphBuffer[np.int8]()
        self.numeric = GraphBuffer[np.float32]()

    def extend(self, obs: list[HeteroObsData]) -> None:
        for o in obs:
            self.boolean.add_single(o.bool)
            self.numeric.add_single(o.float)

    def add_single_dict(self, obs: HeteroObsData) -> None:
        self.boolean.add_single_dict(obs.bool)
        self.numeric.add_single_dict(obs.float)

    @property
    def batch(self) -> HeteroBatchData:
        return HeteroBatchData(
            boolean=batch(list(self.boolean.data)),
            numeric=batch(list(self.numeric.data)),
        )

    def minibatch(self, indices: Iterable[int]) -> HeteroBatchData:
        return HeteroBatchData(
            boolean=batch([self.boolean.data[i] for i in indices]),
            numeric=batch([self.numeric.data[i] for i in indices]),
        )
