from typing import Any

from .stateless_obs_wrapper import create_stateless_wrapper


def remove_none(obs: dict[Any, Any]) -> dict[Any, Any]:
    return {a: v for a, v in obs.items() if v is not None}


RemoveNoneWrapper = create_stateless_wrapper(remove_none)
