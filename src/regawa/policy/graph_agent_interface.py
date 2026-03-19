from abc import ABC, abstractmethod
from pathlib import Path

from torch import Generator as Rngs
from torch import Tensor

from ..data import HeteroIndexedFactorGraph, TorchFactorGraph, TorchHeteroBatchData
from ..model import BaseModel
from .agent_config import AgentConfig
from .types import PolicyOutput


class GraphAgentInterface(ABC):
    @abstractmethod
    def __init__(self, config: AgentConfig, rngs: Rngs, device: str = "cpu"): ...

    @abstractmethod
    def embed(self, data: TorchHeteroBatchData) -> TorchFactorGraph: ...

    @abstractmethod
    def forward(self, actions: Tensor, data: TorchHeteroBatchData) -> PolicyOutput: ...

    @abstractmethod
    def sample_from_obs(
        self,
        obs: HeteroIndexedFactorGraph,
        deterministic: bool = False,
    ) -> PolicyOutput: ...

    @abstractmethod
    def sample(
        self, data: TorchHeteroBatchData, deterministic: bool = False
    ) -> PolicyOutput: ...

    @abstractmethod
    def value(self, data: TorchHeteroBatchData) -> Tensor: ...

    @abstractmethod
    def save_agent(self, path: str | Path, model: BaseModel | None = None): ...

    @abstractmethod
    def num_trainable_params(self) -> int: ...

    @abstractmethod
    def check_compatability(self, model: BaseModel): ...

    @property
    @abstractmethod
    def device(self) -> str: ...

    @device.setter
    @abstractmethod
    def device(self, device: str) -> None: ...

    @property
    @abstractmethod
    def config(self) -> AgentConfig: ...
