from typing import Any, Literal

import gymnasium as gym
import torch
from gymnasium.spaces import MultiDiscrete
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from torch import Generator

from regawa.data.space import n_actions, n_relations, n_types
from regawa.policy.gnn_agent import GraphAgentInterface
from regawa.policy.q_agent.gnn_q_agent import GraphQAgent

from .data.obs import HeteroObsData
from .data.render import to_graphviz
from .data.space import max_arity
from .model import (
    BaseGroundedModel,
    BaseModel,
    Grounding,
    GroundingRange,
    GroundObs,
    ObservableGroundingRange,
    ObservableGroundObs,
)
from .policy import (
    ActionMode,
    AgentConfig,
    GNNParams,
    GraphAgent,
    RecurrentGraphAgent,
    load_agent,
)
from .wrappers import GroundedGraphWrapper, StackingGroundedGraphWrapper

_agent_classes = [GraphAgent, RecurrentGraphAgent, GraphQAgent]
agent_classes = {cls.__name__: cls for cls in _agent_classes}


def agent_config_from_space(
    obs_space: HeteroObsData, action_space: MultiDiscrete, gnn_params: GNNParams
) -> AgentConfig:
    return AgentConfig(
        n_types(obs_space),  # type: ignore
        n_relations(obs_space),  # type: ignore
        n_actions(action_space),  # type: ignore
        arity=max_arity(obs_space),  # type: ignore
        hyper_params=gnn_params,
    )


def agent_from_env(
    agent_class_type: Literal["GraphAgent", "RecurrentGraphAgent", "GraphQAgent"],
    env: gym.Env[HeteroObsData, MultiDiscrete]
    | gym.vector.SyncVectorEnv
    | gym.vector.AsyncVectorEnv,
    params: GNNParams,
    device: str | torch.device = "cpu",
):
    agent_class: type[GraphAgentInterface] = agent_classes[agent_class_type]
    obs_space, action_space = (
        (env.observation_space, env.action_space)
        if not isinstance(env, SyncVectorEnv | AsyncVectorEnv)
        else (env.single_observation_space, env.single_action_space)
    )

    return agent_class(
        agent_config_from_space(obs_space, action_space, params),
        Generator(),
        device=device,
    ).to(device)  # type: ignore


def agent_from_model(
    agent_class: type[GraphAgentInterface],
    model: BaseModel,
    params: GNNParams,
    device: str = "cpu",
) -> GraphAgentInterface:
    n_types = model.num_types
    n_relations = model.num_fluents
    n_actions = model.num_actions
    arity = max_arity(model)

    config = AgentConfig(
        n_types,
        n_relations,
        n_actions,
        arity=arity,
        hyper_params=params,
    )

    rng = Generator()

    return agent_class(config, rng, device).to(device)  # type: ignore


def step_func(agent: GraphAgent, env: gym.Env[Any, Any], deterministic: bool = True):
    def f(
        obs: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action, *_ = agent.sample_from_obs(
            obs,
            deterministic=deterministic,
        )
        next_obs, reward, terminated, truncated, info = env.step(action.squeeze(0))  # type: ignore
        return next_obs, action, reward, terminated, truncated, info

    return f


__all__ = [
    "BaseModel",
    "BaseGroundedModel",
    "GroundedGraphWrapper",
    "StackingGroundedGraphWrapper",
    "Grounding",
    "GNNParams",
    "ActionMode",
    "AgentConfig",
    "GraphAgent",
    "RecurrentGraphAgent",
    "agent_from_env",
    "agent_from_model",
    "to_graphviz",
    "ObservableGroundObs",
    "ObservableGroundingRange",
    "GroundObs",
    "GroundingRange",
    "load_agent",
    "GraphAgentInterface",
]
