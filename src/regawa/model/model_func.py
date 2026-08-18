"""Utility functions for models"""

import json
from collections.abc import Callable
from functools import cache

from regawa.model.null import NullConst

from . import check_model
from .base_model import BaseModel
from .generic_model import GenericModel


def max_fluent_arity(model: BaseModel):
    return max(model.arity(fluent) for fluent in model.fluents)  # type: ignore


def fn_type_attributes(model: BaseModel) -> Callable[[str], tuple[str, ...]]:
    """
    Returns a function that takes an object type and returns a tuple of attribute names for that type
    """
    vp = model.variable_params  # type: ignore
    d: dict[str, tuple[str, ...]] = {
        value[0]: tuple([k for k, v in vp.items() if v == value])  # type: ignore
        for _, value in vp.items()  # type: ignore
        if len(value) == 1  # type: ignore
    }

    @cache
    def type_attributes(object_type: str) -> str:
        return d[object_type]  # type: ignore

    return type_attributes


def fn_fluents_of_arity(model: BaseModel) -> Callable[[int], tuple[str, ...]]:
    """
    Returns a function that takes an arity and returns a tuple of fluents with that arity.
    """
    arities = {model.arity(fluent) for fluent in model.fluents}  # type: ignore

    d: dict[int, tuple[str, ...]] = {
        arity: tuple([f for f in model.fluents if model.arity(f) == arity])
        for arity in arities
    }

    @cache
    def fluents_of_arity(arity: int) -> tuple[str, ...]:
        return d[arity]

    return fluents_of_arity


"""
The two valid_* function use different criteria for determining whether an action fluent is valid for an object type.
fn_valid_action_fluents_given_type checks if the object type is in the fluent's parameter types,
while fn_valid_action_fluents_given_arity checks if the fluent has arity greater than 0 (i.e. it takes at least one object parameter).
"""


def fn_valid_action_fluents_given_type(
    model: BaseModel,
) -> Callable[[str], tuple[bool, ...]]:
    """
    Returns a function that takes an object type and returns a tuple of booleans indicating
    whether each action fluent is valid for that object based on the fluent's parameter types.
    The null object is assumed to be a valid object type for all fluents.
    """

    @cache
    def is_valid(fluent: str, o_t: str) -> bool:
        return (
            o_t in model.fluent_params(fluent) if model.arity(fluent) > 0 else True
        )  # assume fluents with 0 arity are valid for all object types

    @cache
    def valid_action_fluents_given_type(obj_type: str) -> tuple[bool, ...]:
        return tuple(is_valid(fluent, obj_type) for fluent in model.action_fluents)

    return valid_action_fluents_given_type


def fn_valid_action_fluents_given_arity(
    model: BaseModel,
) -> Callable[[str], tuple[bool, ...]]:
    """
    Returns a function that takes an object type and returns a tuple of booleans indicating
    whether each action fluent in the model is valid for that object based on the arity of the fluent.
    Nullary predicates do not take any object parameters, so they are never valid for any object type, except the null object.
    """

    @cache
    def is_valid(fluent: str, o_t: str) -> bool:
        return (
            True
            if model.arity(fluent) > 0
            # Assume NULL_TYPE is the only valid object type for 0-arity fluents
            else o_t == NullConst.type
        )

    @cache
    def valid_action_fluents_given_arity(obj_type: str) -> tuple[bool, ...]:
        return tuple(is_valid(fluent, obj_type) for fluent in model.action_fluents)

    return valid_action_fluents_given_arity


def model_to_json(model: BaseModel) -> str:


    try:
        check_model(model)
    except Exception as e:
        raise ValueError(f"Model does not pass check: {e}") from e

    model_dict = {
        "types": model.types,
        "fluents": model.fluents,
        "action_fluents": model.action_fluents,
        "fluent_ranges": {f: model.fluent_range(f).__name__ for f in model.fluents},
        "fluent_params": {f: model.fluent_params(f) for f in model.fluents},
    }
    return json.dumps(model_dict, indent=4)


def model_from_json(model_json: str) -> BaseModel:
    return GenericModel.from_json(model_json)
