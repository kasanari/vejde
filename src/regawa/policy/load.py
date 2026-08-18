import json
import zipfile
from pathlib import Path

import torch

from regawa.model.base_model import BaseModel
from regawa.model.generic_model import GenericModel

from .agent_config import ActionMode, AgentConfig, BooleanEmbedderType, GNNParams
from .gnn_agent import GraphAgent
from .recurrent_gnn_agent import RecurrentGraphAgent
from .save import SaveFormat, activation_to_str

## TODO check that the size of the loaded model matches the basemodel used

str_to_activation = {v: k for k, v in activation_to_str.items()}


def load_agent[T: GraphAgent | RecurrentGraphAgent](
    cls: type[T],
    path: str | Path,
    device: str = "cpu",
    loadformat: SaveFormat = SaveFormat.ZIP,
) -> tuple[T, AgentConfig, BaseModel | None]:
    if loadformat == SaveFormat.ZIP:
        return load_agent_from_zip(cls, path, device)

    agent, config = load_agent_legacy(cls, path, device)
    return agent, config, None


def load_agent_legacy[T: GraphAgent | RecurrentGraphAgent](
    cls: type[T], path: str | Path, device: str = "cpu"
) -> tuple[T, AgentConfig]:
    data = torch.load(path, weights_only=False, map_location=device)  # type: ignore

    data["config"]["hyper_params"] = GNNParams(**data["config"]["hyper_params"])

    config = AgentConfig(**data["config"])
    agent = cls(config, None, device=device)
    agent.load_state_dict(data["state_dict"])

    agent.device = device

    return agent, config


def load_agent_from_zip[T: GraphAgent | RecurrentGraphAgent](
    cls: type[T],
    path: str,
    device: str = "cpu",
) -> tuple[T, AgentConfig, BaseModel | None]:
    with (
        zipfile.ZipFile(path, "r") as z,
        z.open("agent.pt") as f,
        z.open("config.json", "r") as cf,
    ):
        state_dict = torch.load(f, weights_only=True, map_location=device)
        config_dict = json.loads(cf.read().decode("utf-8"))

        # check if model.json exists
        if "model.json" in z.namelist():
            with z.open("model.json", "r") as mf:
                model_json = mf.read().decode("utf-8")
            model = GenericModel.from_json(model_json)
        else:
            model = None

    hyper_params = config_dict["hyper_params"]
    hyper_params["action_mode"] = ActionMode(ActionMode[hyper_params["action_mode"]])
    hyper_params["activation"] = str_to_activation[hyper_params["activation"]]()
    hyper_params["boolean_embedder_type"] = BooleanEmbedderType[
        hyper_params["boolean_embedder_type"]
    ]

    config_dict["hyper_params"] = GNNParams(**hyper_params)

    config = AgentConfig(**config_dict)
    agent = cls(config, None, device=device)
    agent.load_state_dict(state_dict)

    agent.device = device

    return agent, config, model
