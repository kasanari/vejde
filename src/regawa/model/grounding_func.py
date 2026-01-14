import logging
from collections.abc import Callable, Sequence
from functools import cache

import numpy as np

from .base_grounded_model import Grounding, GroundObs
from .null import NullConst

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


def has_valid_parameters(
    action: Grounding,
    obj_to_type: Callable[[str], str],  # maps object to its type
    fluent_params: Callable[
        [str], tuple[str, ...]
    ],  # maps fluent to its parameter types
) -> bool:
    """
    Checks if the parameters of an action are valid based on the fluent's parameter types.
    """
    action_fluent = predicate(action)
    param_types = fluent_params(action_fluent)
    params: tuple[str, ...] = objects(action)

    if len(param_types) != len(params):
        return False

    for intended_param, param in zip(param_types, params, strict=False):
        if intended_param != obj_to_type(param):
            return False

    return True


def to_dict_action(
    action: Grounding,
    obj_to_type: Callable[[str], str],
    fluent_params: Callable[[str], tuple[str, ...]],
) -> GroundObs:
    """
    Converts an action (Grounding) to a dictionary representation. Going from (predicate, obj1, obj2) to {(predicate, obj1, obj2): True}.
    If the action has invalid parameters, it is converted to a no-op action (i.e. "NOP" predicate).
    No-op actions are represented as an empty dictionary.
    """
    action_fluent = predicate(action)
    num_params = len(fluent_params(action_fluent))
    action_arity = len(fluent_params(action_fluent))

    if action_fluent == NullConst.action:
        return {}

    if action_arity == 0:
        return {(action_fluent,): np.bool_(True)}

    has_valid_param = has_valid_parameters(action, obj_to_type, fluent_params)
    action_fluent = NullConst.action if not has_valid_param else action_fluent

    if not has_valid_param:
        logger.warning(f"Invalid parameters for action {action}")

    a = (
        (action_fluent, *objects(action)[:num_params])
        if has_valid_param
        else (NullConst.action, NullConst.id)
    )

    return {} if action_fluent == NullConst.action else {a: np.bool_(True)}


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
