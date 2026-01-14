from regawa.model.grounding_func import remove_false

from .stateless_obs_wrapper import create_stateless_wrapper

RemoveFalseWrapper = create_stateless_wrapper(remove_false)
