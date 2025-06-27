from typing import Any
from torch import Generator
from gymnasium.spaces import MultiDiscrete
from regawa.data.data import HeteroObsData
from .policy import ActionMode, GNNParams, AgentConfig
from .model import GroundValue
from .model import BaseGroundedModel
from .model import BaseModel
from .wrappers import StackingGroundedGraphWrapper
from .wrappers import GroundedGraphWrapper
from .policy import GraphAgent
from .wrappers import gym_utils
from .model import max_arity
import gymnasium as gym


def agent_from_env(
    env: gym.Env[HeteroObsData, MultiDiscrete], params: GNNParams, device: str = "cpu"
):
    n_types = gym_utils.n_types(env.observation_space)
    n_relations = gym_utils.n_relations(env.observation_space)
    n_actions = gym_utils.n_actions(env.action_space)

    config = AgentConfig(
        n_types,
        n_relations,
        n_actions,
        remove_false_fluents=True,
        arity=gym_utils.max_arity(env.observation_space),
        hyper_params=params,
    )

    rng = Generator()

    return GraphAgent(config, rng, device)


def agent_from_model(model: BaseModel, params: GNNParams, device: str = "cpu"):
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

    return GraphAgent(config, rng, device)


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
    "GroundValue",
    "GNNParams",
    "ActionMode",
    "AgentConfig",
    "GraphAgent",
    "agent_from_env",
    "agent_from_model",
]
