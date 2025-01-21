from typing import Any
from pyRDDLGym.core.compiler.model import RDDLLiftedModel  # type: ignore
from functools import cache


def get_groundings(model: RDDLLiftedModel, fluents: dict[str, Any]) -> set[str]:
    return set(
        dict(model.ground_vars_with_values(fluents)).keys()  # type: ignore
    )


@cache
def split_rddl_grounding(grounding: str):
    pred, *args = grounding.split("___")
    args = args[0].split("__") if args else ()
    return (pred, *args)
