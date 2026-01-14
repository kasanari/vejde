import logging
from collections.abc import Callable, Sequence
from functools import cache

import numpy as np

from .base_grounded_model import Grounding, GroundObs

logger = logging.getLogger(__name__)


@cache
def objects(key: Grounding) -> tuple[str, ...]:
    return key[1:]


@cache
def predicate(key: Grounding) -> str:
    return key[0]


@cache
def arity(grounding: Grounding) -> int:
    o = objects(grounding)
    return len(o)


def num_edges(groundings: list[Grounding], arities: Callable[[str], int]) -> int:
    return sum(arities(predicate(g)) for g in groundings)


def fn_is_numeric(fluent_range: Callable[[str], type]):
    """
    Returns a function that takes a grounding and returns whether it is numeric (int or float).
    """

    @cache
    def is_numeric(g: Grounding):
        return fluent_range(predicate(g)) is float or fluent_range(predicate(g)) is int

    return is_numeric


def numeric_groundings(
    groundings: Sequence[Grounding], is_numeric: Callable[[Grounding], bool]
) -> list[Grounding]:
    """
    Returns a list of numeric groundings from the given list of groundings.
    """
    return [g for g in groundings if is_numeric(g)]


def fn_is_bool(fluent_range: Callable[[str], type]):
    """
    Returns a function that takes a grounding and returns whether it is boolean.
    """

    @cache
    def is_bool(g: Grounding):
        return fluent_range(predicate(g)) is bool

    return is_bool


def bool_groundings(
    groundings: Sequence[Grounding], is_bool: Callable[[Grounding], bool]
) -> list[Grounding]:
    """
    Returns a list of boolean groundings from the given list of groundings.
    """
    return [g for g in groundings if is_bool(g)]


def filter_none_groundings(rddl_obs: GroundObs) -> GroundObs:
    filtered_groundings = [
        g
        for g in rddl_obs
        if rddl_obs[g] is not None  # type: ignore
    ]

    filtered_obs: GroundObs = {k: rddl_obs[k] for k in filtered_groundings}
    return filtered_obs


def remove_false(obs: GroundObs) -> GroundObs:
    return {a: v for a, v in obs.items() if v is not False and v is not np.bool_(False)}
