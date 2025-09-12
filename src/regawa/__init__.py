from typing import Any
from torch import Generator
from gymnasium.spaces import MultiDiscrete

from regawa.policy.gnn_agent import GraphAgentInterface
from .data import HeteroObsData
from .policy import ActionMode, GNNParams, AgentConfig
from .model import (
    Grounding,
    GroundObs,
    GroundingRange,
    ObservableGroundingRange,
    ObservableGroundObs,
)
from .model import BaseGroundedModel
from .model import BaseModel
from .wrappers import StackingGroundedGraphWrapper
from .wrappers import GroundedGraphWrapper
from .policy import GraphAgent, RecurrentGraphAgent
from .wrappers import gym_utils
from .model import max_arity
import gymnasium as gym
from .wrappers.render_utils import to_graphviz
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv


def agent_from_env(
    agent_class: type[GraphAgentInterface],
    env: gym.Env[HeteroObsData, MultiDiscrete]
    | gym.vector.SyncVectorEnv
    | gym.vector.AsyncVectorEnv,
    params: GNNParams,
    device: str = "cpu",
):
    obs_space, action_space = (
        (env.observation_space, env.action_space)
        if not isinstance(env, (SyncVectorEnv, AsyncVectorEnv))
        else (env.single_observation_space, env.single_action_space)
    )

    n_types = gym_utils.n_types(obs_space)  # type: ignore
    n_relations = gym_utils.n_relations(obs_space)  # type: ignore
    n_actions = gym_utils.n_actions(action_space)  # type: ignore

    config = AgentConfig(
        n_types,
        n_relations,
        n_actions,
        remove_false_fluents=True,
        arity=gym_utils.max_arity(obs_space),  # type: ignore
        hyper_params=params,
    )

    rng = Generator()

    return agent_class(config, rng, device=device)


def agent_from_model(
    agent_class: type[GraphAgentInterface],
    model: BaseModel,
    params: GNNParams,
    device: str = "cpu",
):
    n_types = model.num_types
    n_relations = model.num_fluents
    n_actions = model.num_actions
    arity = max_arity(model)

    config = AgentConfig(
        n_types,
        n_relations,
        n_actions,
        remove_false_fluents=True,
        arity=arity,
        hyper_params=params,
    )

    rng = Generator()

    return agent_class(config, rng, device)


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
]
