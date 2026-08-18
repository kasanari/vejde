import tempfile

import torch as th

from regawa import (
    ActionMode,
    GNNParams,
    GraphAgent,
    agent_from_model,
    load_agent,
    save_agent,
)
from regawa.model.base_model import BaseModel

params = GNNParams(
    layers=4,
    embedding_dim=4,
    activation=th.nn.Mish(),
    aggregation="max",
    action_mode=ActionMode.NODE_THEN_ACTION,
)


def test_save_load_agent(test_model: BaseModel):
    agent = agent_from_model(GraphAgent, test_model, params, device="cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/agent.zip"
        save_agent(agent, agent.config, path, model=test_model)

        loaded_agent, config, model = load_agent(GraphAgent, path, device="cpu")
    assert isinstance(loaded_agent, GraphAgent)
    assert config.hyper_params == agent.config.hyper_params
    assert loaded_agent.config == agent.config
    assert model is not None
    assert model.num_types == test_model.num_types
    assert model.num_fluents == test_model.num_fluents
    assert model.num_actions == test_model.num_actions
    assert model.fluents == test_model.fluents
    assert config == agent.config
