from gymnasium.spaces import Dict, MultiDiscrete
from .space import HeteroStateSpace
from gymnasium.vector.utils.space_utils import batch_differing_spaces  # type: ignore


def max_arity(observation_space: Dict) -> int:
    return observation_space.bool.edge_attr.feature_space.n - 1  # type: ignore


def n_types(observation_space: HeteroStateSpace):
    return int(observation_space.bool.factor.feature_space.n)  # type: ignore


def n_relations(observation_space: HeteroStateSpace):
    return int(observation_space.bool.var.var_type.feature_space.n)  # type: ignore


def n_actions(action_space: MultiDiscrete):
    return int(action_space.nvec[0])


@batch_differing_spaces.register(HeteroStateSpace)  # type: ignore
def batch_differing_spaces(spaces: list[HeteroStateSpace]):
    return spaces
