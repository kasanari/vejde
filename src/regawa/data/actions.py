import numpy as np
from numpy.typing import NDArray


from typing import NamedTuple


class ActionMask(NamedTuple):
    # mask that indicates which actions are valid for each factor, given the predicate type. Length matches factor.
    action_type_mask: NDArray[np.bool_]
    # mask that indicates which actions are valid for each factor, given the predicate arity. Objects are not valid for predicates with no arguments. Length matches factor.
    action_arity_mask: NDArray[np.bool_]
