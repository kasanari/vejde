from functools import cache

from .base_model import BaseModel


def max_arity(model: BaseModel):
    return max(model.arity(fluent) for fluent in model.fluents)  # type: ignore


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


def valid_action_fluents(model: BaseModel):
    @cache
    def is_valid(fluent: str, o_t: str) -> bool:
        return (
            o_t in model.fluent_params(fluent)
            if model.arity(fluent) > 0
            else o_t
            == "None"  # Assume "None" is a valid object type for 0-arity fluents
        )

    def f(obj_type: str) -> tuple[bool, ...]:
        return tuple(is_valid(fluent, obj_type) for fluent in model.action_fluents)

    return f
