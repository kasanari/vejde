from functools import cache

from .base_model import BaseModel


def max_arity(model: BaseModel):
    return max(model.arity(fluent) for fluent in model.fluents)  # type: ignore


def fn_type_attributes(model: BaseModel):
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


def fn_fluents_of_arity(model: BaseModel):
    """
    Returns a function that takes an arity and returns a tuple of fluents with that arity.
    """
    arities = set(model.arity(fluent) for fluent in model.fluents)  # type: ignore

    d: dict[int, tuple[str, ...]] = {
        arity: tuple([f for f in model.fluents if model.arity(f) == arity])
        for arity in arities
    }

    @cache
    def fluents_of_arity(arity: int) -> tuple[str, ...]:
        return d[arity]

    return fluents_of_arity


def fn_valid_action_fluents_given_type(model: BaseModel):
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


def fn_valid_action_fluents_given_arity(model: BaseModel):
    """
    Returns a function that takes an object type and returns a tuple of booleans indicating
    whether each action fluent is valid for that object based on the arity of the predicate.
    Nullary predicates do not take any object parameters, so they are never valid for any object type, except the null object.
    The null object is assumed to be a valid object type for all fluents.
    """
    @cache
    def is_valid(fluent: str, o_t: str) -> bool:
        return (
            True
            if model.arity(fluent) > 0
            else o_t
            == "None"  # Assume "None" is a valid object type for 0-arity fluents
        )

    @cache
    def valid_action_fluents_given_arity(obj_type: str) -> tuple[bool, ...]:
        return tuple(is_valid(fluent, obj_type) for fluent in model.action_fluents)

    return valid_action_fluents_given_arity
