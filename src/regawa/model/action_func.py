import random
from collections.abc import Callable, Sequence

import numpy as np
from gymnasium.spaces import Dict

from regawa.model import (
    Grounding,
    GroundObs,
)
from regawa.model.grounding_func import logger, objects, predicate
from regawa.model.null import NullConst


def from_dict_action(
    action: tuple[str, ...],
    action_to_idx: Callable[[str], int],
    obj_to_idx: Callable[[str], int],
) -> tuple[int, ...]:
    action_idx = action_to_idx(action[0])
    object_idxs = [obj_to_idx(obj) for obj in action[1:]]
    return (action_idx, *object_idxs)


def idx_action_to_ground_value(
    action: Sequence[int],
    idx_to_action: Callable[[int], str],
    idx_to_obj: Callable[[int], str],
) -> Grounding:
    action_name = idx_to_action(action[0])
    o = tuple(idx_to_obj(obj_idx) for obj_idx in action[1:])
    return (action_name, *o)


def sample_action(action_space: Dict) -> dict[str, int]:
    action = action_space.sample()  # type: ignore
    chosen_action, value = random.choice(list(action.items()))  # type: ignore
    return {chosen_action: value}


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
