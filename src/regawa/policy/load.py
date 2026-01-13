from typing import TypeVar

import torch

from .agent_config import AgentConfig, GNNParams
from .gnn_agent import GraphAgent
from .recurrent_gnn_agent import RecurrentGraphAgent

T = TypeVar("T", bound=GraphAgent | RecurrentGraphAgent)

## TODO check that the size of the loaded model matches the basemodel used


def load_agent(cls: type[T], path: str, device: str = "cpu") -> tuple[T, AgentConfig]:
    data = torch.load(path, weights_only=False, map_location=device)  # type: ignore

    data["config"]["hyper_params"] = GNNParams(**data["config"]["hyper_params"])

    config = AgentConfig(**data["config"])
    agent = cls(config, None, device=device)
    agent.load_state_dict(data["state_dict"])

    agent.device = device

    return agent, config
