from dataclasses import asdict
from typing import Any
import torch
import torch.nn as nn
from .agent_utils import AgentConfig


def save_agent(agent: nn.Module, config: AgentConfig, path: str):
    state_dict = agent.state_dict()
    to_save: dict[str, Any] = {}
    to_save["config"] = asdict(config)
    to_save["state_dict"] = state_dict
    torch.save(to_save, path)  # type: ignore
