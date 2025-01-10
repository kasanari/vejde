from .base_model import BaseModel
from functools import cache
from gymnasium.spaces import Dict, MultiDiscrete


def type_attributes(model: BaseModel):
    vp = model.variable_params  # type: ignore
    d: dict[str, tuple[str, ...]] = {
        value[0]: tuple([k for k, v in vp.items() if v == value])  # type: ignore
        for _, value in vp.items()  # type: ignore
        if len(value) == 1  # type: ignore
    }

    @cache
    def f(object_type: str) -> str:
        return d[object_type]  # type: ignore

    return f


def fluents_of_arity(model: BaseModel):
    arities = set(model.arity(fluent) for fluent in model.fluents)  # type: ignore

    d: dict[int, tuple[str, ...]] = {
        arity: tuple([f for f in model.fluents if model.arity(f) == arity])
        for arity in arities
    }

    @cache
    def f(arity: int) -> tuple[str, ...]:
        return d[arity]

    return f


def n_types(observation_space: Dict):
    return int(observation_space["factor"].feature_space.n)  # type: ignore


def n_relations(observation_space: Dict):
    return int(observation_space["var_type"].feature_space.n)  # type: ignore


def n_actions(action_space: MultiDiscrete):
    return int(action_space.nvec[0])  # type: ignore
