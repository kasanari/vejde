from typing import Any, TypeVar

import gymnasium as gym
import torch
from gymnasium.spaces import MultiDiscrete
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from torch import Generator

from .data.obs import HeteroIndexedFactorGraph
from .data.render import to_graphviz
from .data.space import max_arity, n_actions, n_relations, n_types
from .model import (
    BaseGroundedModel,
    BaseModel,
    Grounding,
    GroundingRange,
    GroundObs,
    ObservableGroundingRange,
    ObservableGroundObs,
    max_fluent_arity,
)
from .policy import (
    ActionMode,
    AgentConfig,
    GNNParams,
    GraphAgent,
    GraphAgentInterface,
    RecurrentGraphAgent,
    load_agent,
    save_agent,
)
from .wrappers import GroundedGraphWrapper, StackingGroundedGraphWrapper


def agent_config_from_space(
    obs_space: HeteroIndexedFactorGraph,
    action_space: MultiDiscrete,
    gnn_params: GNNParams,
) -> AgentConfig:
    return AgentConfig(
        n_types(obs_space),  # type: ignore
        n_relations(obs_space),  # type: ignore
        n_actions(action_space),  # type: ignore
        arity=max_arity(obs_space),  # type: ignore
        hyper_params=gnn_params,
    )


V = TypeVar("V", bound=GraphAgentInterface)


def agent_from_env(
    agent_class: V,
    env: gym.Env[HeteroIndexedFactorGraph, MultiDiscrete]
    | gym.vector.SyncVectorEnv
    | gym.vector.AsyncVectorEnv,
    params: GNNParams,
    device: str | torch.device = "cpu",
    rng: Generator | None = None,
) -> V:
    obs_space, action_space = (
        (env.observation_space, env.action_space)
        if not isinstance(env, SyncVectorEnv | AsyncVectorEnv)
        else (env.single_observation_space, env.single_action_space)
    )

    return agent_class(
        agent_config_from_space(obs_space, action_space, params),
        rngs=rng if rng is not None else Generator(),
        device=device,
    ).to(device)  # type: ignore


def agent_from_model(
    agent_class: type[V],
    model: BaseModel,
    params: GNNParams,
    device: str = "cpu",
    rng: Generator | None = None,
) -> V:
    n_types = model.num_types
    n_relations = model.num_fluents
    n_actions = model.num_actions
    arity = max_fluent_arity(model)

    config = AgentConfig(
        n_types,
        n_relations,
        n_actions,
        arity=arity,
        hyper_params=params,
    )

    rng = rng if rng is not None else Generator()

    return agent_class(config, rngs=rng, device=device).to(device)  # type: ignore


def step_func(
    agent: GraphAgentInterface, env: gym.Env[Any, Any], rng: Generator | None
):
    def f(
        obs: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action, *_ = agent.sample_from_obs(
            obs,
            rng=rng,
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
    "save_agent",
    "GraphAgentInterface",
]
