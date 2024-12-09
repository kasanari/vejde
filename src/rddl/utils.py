from typing import Any
from pyRDDLGym.core.compiler.model import RDDLLiftedModel  # type: ignore


def get_groundings(model: RDDLLiftedModel, fluents: dict[str, Any]) -> set[str]:
    return set(
        dict(model.ground_vars_with_values(fluents)).keys()  # type: ignore
    )
