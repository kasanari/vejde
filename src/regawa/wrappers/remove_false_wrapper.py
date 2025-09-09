import numpy as np
from regawa import GroundObs
from regawa.wrappers.stateless_obs_wrapper import create_stateless_wrapper


def remove_false(obs: GroundObs) -> GroundObs:
    return {a: v for a, v in obs.items() if v is not False and v is not np.bool_(False)}


RemoveFalseWrapper = create_stateless_wrapper(remove_false)
