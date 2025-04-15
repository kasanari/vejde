from .gnn.agent_utils import ActionMode, GNNParams, AgentConfig
from .model import GroundValue
from .model.base_grounded_model import BaseGroundedModel
from .model.base_model import BaseModel
from .wrappers.pomdp_wrapper import StackingGroundedGraphWrapper
from .wrappers.wrapper import GroundedGraphWrapper
from .gnn import GraphAgent
from torch import Generator
from .wrappers import gym_utils
from .model.utils import max_arity
import gymnasium as gym


def agent_from_env(env: gym.Env, params: GNNParams, device: str = "cpu"):
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
