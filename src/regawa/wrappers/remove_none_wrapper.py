from regawa import GroundObs, ObservableGroundObs

from .stateless_obs_wrapper import create_stateless_wrapper


def remove_none(obs: ObservableGroundObs) -> GroundObs:
    return {a: v for a, v in obs.items() if v is not None}


RemoveNoneWrapper = create_stateless_wrapper(remove_none)
