from typing import TypeVar
from .gnn_agent import GraphAgent
from .recurrent_gnn_agent import RecurrentGraphAgent
from .agent_utils import AgentConfig, GNNParams
import torch

T = TypeVar("T", bound=GraphAgent | RecurrentGraphAgent)

## TODO check that the size of the loaded model matches the basemodel used


def load_agent(cls: type[T], path: str, device: str = "cpu") -> tuple[T, AgentConfig]:
    data = torch.load(path, weights_only=False, map_location=device)  # type: ignore

    data["config"]["hyper_params"] = GNNParams(**data["config"]["hyper_params"])

    if "remove_false_fluents" not in data["config"]:
        data["config"]["remove_false_fluents"] = False  # for backward compatibility

    config = AgentConfig(**data["config"])
    agent = cls(config, None)
    agent.load_state_dict(data["state_dict"])

    return agent, config
