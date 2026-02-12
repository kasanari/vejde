import json
import tempfile
import zipfile
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from regawa.model.base_model import BaseModel
from regawa.model.model_func import model_to_json

from .agent_config import AgentConfig

activation_to_str = {
    nn.ReLU: "relu",
    nn.GELU: "gelu",
    nn.SiLU: "silu",
    nn.Mish: "mish",
    nn.LeakyReLU: "leaky_relu",
}


class Encoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Enum):
            return o.name
        if isinstance(o, nn.Module):
            act_type = type(o)
            if act_type in activation_to_str:
                return activation_to_str[act_type]
            return str(o)
        if isinstance(o, np.int64):
            return int(o)
        return super().default(o)


class SaveFormat(Enum):
    PT = "pt"
    ZIP = "zip"


def save_agent(
    agent: nn.Module,
    config: AgentConfig,
    path: str | Path,
    saveformat: SaveFormat = SaveFormat.ZIP,
    model: BaseModel | None = None,
):
    if saveformat == SaveFormat.ZIP:
        save_agent_as_zip(agent, config, path, model)
    else:
        save_agent_legacy(agent, config, path)


def save_agent_legacy(agent: nn.Module, config: AgentConfig, path: str | Path):
    state_dict = agent.state_dict()
    to_save: dict[str, Any] = {}
    to_save["config"] = asdict(config)
    to_save["state_dict"] = state_dict
    torch.save(to_save, path)  # type: ignore


def save_agent_as_zip(
    agent: nn.Module,
    config: AgentConfig,
    path: str | Path,
    model: BaseModel | None = None,
):
    """Saves the agent and optionally the model as a zip file."""
    state_dict = agent.state_dict()
    config_json = json.dumps(asdict(config), cls=Encoder)

    temp_path = Path(tempfile.mkstemp(suffix=".pt")[1])
    torch.save(state_dict, temp_path)

    with zipfile.ZipFile(path, "w") as zipf:
        zipf.write(temp_path, arcname="agent.pt")
        zipf.writestr("config.json", config_json)
        if model is not None:
            model_json = model_to_json(model)
            zipf.writestr("model.json", model_json)

    temp_path.unlink()
