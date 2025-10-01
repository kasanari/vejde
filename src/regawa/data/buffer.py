import numpy as np
from regawa.data.batch import BatchData, HeteroBatchData, batch
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
        self.buffers = {
            "bool": GraphBuffer[np.bool_](),
            "float": GraphBuffer[np.float32](),
        }

    def extend(self, obs: list[HeteroObsData]) -> None:
        for o in obs:
            for t in self.buffers:
                self.buffers[t].add_single(o.__getattribute__(t))

    def __iter__(self):
        return self.buffers.__iter__()

    def add_single_dict(self, obs: HeteroObsData) -> None:
        for t in self.buffers:
            self.buffers[t].add_single_dict(obs.__getattribute__(t))

    @property
    def batch(self) -> HeteroBatchData:
        return HeteroBatchData(
            boolean=batch(list(self.buffers["bool"].data)),
            numeric=batch(list(self.buffers["float"].data)),
        )

    def minibatch(self, indices: Iterable[int]) -> HeteroBatchData:
        return HeteroBatchData(
            boolean=batch([self.buffers["bool"].data[i] for i in indices]),
            numeric=batch([self.buffers["float"].data[i] for i in indices]),
        )
