from typing import Any
import numpy as np
from regawa.wrappers.stateless_obs_wrapper import create_stateless_wrapper


def remove_false(obs: dict[Any, Any]) -> dict[Any, Any]:
    return {a: v for a, v in obs.items() if v is not False and v is not np.bool_(False)}


RemoveFalseWrapper = create_stateless_wrapper(remove_false)
