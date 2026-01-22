from .space import HeteroStateSpace, FactorGraphSpace
from .space_func import max_arity, n_actions, n_relations, n_types

__all__ = [
    "n_actions",
    "n_relations",
    "n_types",
    "HeteroStateSpace",
    "FactorGraphSpace",
    "max_arity",
]
