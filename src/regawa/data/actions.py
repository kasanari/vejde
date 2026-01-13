from collections.abc import Callable, Sequence
from gymnasium.spaces import Dict
import numpy as np
from numpy.typing import NDArray

import random
from typing import NamedTuple

from regawa.model import BaseModel, Grounding
from regawa.model.utils import (
    fn_valid_action_fluents_given_arity,
    fn_valid_action_fluents_given_type,
)


class ActionMask(NamedTuple):
    # mask that indicates which actions are valid for each factor, given the predicate type. Length matches factor.
    action_type_mask: NDArray[np.bool_]
    # mask that indicates which actions are valid for each factor, given the predicate arity. Objects are not valid for predicates with no arguments. Length matches factor.
    action_arity_mask: NDArray[np.bool_]


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


def fn_action_masks(
    model: BaseModel,
):
    action_fluent_type_mask = fn_valid_action_fluents_given_type(model)
    action_fluent_arity_mask = fn_valid_action_fluents_given_arity(model)

    def f(
        object_types: Sequence[str],
    ) -> ActionMask:
        return ActionMask(
            np.array(
                tuple(map(action_fluent_type_mask, object_types)),
                dtype=np.bool_,
            ),  # n_object x n_actions
            np.array(
                tuple(map(action_fluent_arity_mask, object_types)),
                dtype=np.bool_,
            ),  # n_object x n_actions
        )

    return f
