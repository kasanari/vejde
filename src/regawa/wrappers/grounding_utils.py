from functools import cache
import itertools
import logging
from collections import deque
from collections.abc import Callable

import numpy as np

from regawa.model import GroundValue
from regawa.wrappers.util_types import Edge, Object

logger = logging.getLogger(__name__)


@cache
def objects(key: GroundValue) -> tuple[str, ...]:
    return key[1:]


@cache
def predicate(key: GroundValue) -> str:
    return key[0]


def fn_objects_with_type(relation_to_types: Callable[[str, int], str]):
    @cache
    def objects_with_type(
        key: GroundValue,
    ) -> list[Object]:
        p = predicate(key)
        os = objects(key)
        return [Object(o, relation_to_types(p, i)) for i, o in enumerate(os)]

    return objects_with_type


@cache
def arity(grounding: GroundValue) -> int:
    o = objects(grounding)
    return len(o)


def has_valid_parameters(
    action: GroundValue,
    obj_to_type: Callable[[str], str],
    fluent_params: Callable[[str], tuple[str, ...]],
) -> bool:
    action_fluent = predicate(action)
    param_types = fluent_params(action_fluent)
    params: tuple[str, ...] = objects(action)

    if len(param_types) != len(params):
        return False

    for intended_param, param in zip(param_types, params):
        if intended_param != obj_to_type(param):
            return False

    return True


def to_dict_action(
    action: GroundValue,
    obj_to_type: Callable[[str], str],
    fluent_params: Callable[[str], tuple[str, ...]],
) -> dict[GroundValue, np.bool_]:
    action_fluent = predicate(action)
    action_arity = len(fluent_params(action_fluent))
    if action_arity == 0:
        return {} if action_fluent == "None" else {(action_fluent,): np.bool_(True)}

    has_valid_param = has_valid_parameters(action, obj_to_type, fluent_params)
    action_fluent = "None" if not has_valid_param else action_fluent

    num_params = len(fluent_params(action_fluent))

    if not has_valid_param:
        logger.warning(f"Invalid parameters for action {action}")

    a = (action_fluent, *objects(action)[:num_params])

    action_dict = {} if action_fluent == "None" else {a: np.bool_(True)}

    return action_dict


@cache
def get_edges(key: GroundValue) -> list[Edge]:
    return [Edge(key, object, pos) for pos, object in enumerate(objects(key))]


def create_edges(d: list[GroundValue]) -> list[Edge]:
    edges = [get_edges(key) for key in d]
    return list(itertools.chain(*edges))


def num_edges(groundings: list[GroundValue], arities: Callable[[str], int]) -> int:
    return sum(arities(predicate(g)) for g in groundings)


def fn_is_numeric(fluent_range: Callable[[str], type]):
    @cache
    def is_numeric(g: GroundValue):
        return fluent_range(predicate(g)) is float or fluent_range(predicate(g)) is int

    return is_numeric


def numeric_groundings(
    groundings: list[GroundValue], is_numeric: Callable[[GroundValue], bool]
) -> list[GroundValue]:
    return [g for g in groundings if is_numeric(g)]


def fn_is_bool(fluent_range: Callable[[str], type]):
    @cache
    def is_bool(g: GroundValue):
        return fluent_range(predicate(g)) is bool

    return is_bool


def bool_groundings(
    groundings: list[GroundValue], is_bool: Callable[[GroundValue], bool]
) -> list[GroundValue]:
    return [g for g in groundings if is_bool(g)]
