from dataclasses import asdict
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from .agent_config import AgentConfig


def save_agent(agent: nn.Module, config: AgentConfig, path: str | Path):
    state_dict = agent.state_dict()
    to_save: dict[str, Any] = {}
    to_save["config"] = asdict(config)
    to_save["state_dict"] = state_dict
    torch.save(to_save, path)  # type: ignore
