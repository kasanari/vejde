from functools import cached_property

from regawa.model import GroundValue

from .rddl_grounded_model import RDDLGroundedModel
from .rddl_utils import get_groundings, rddl_ground_to_tuple


class RDDLPOMDPGroundedModel(RDDLGroundedModel):
    @cached_property
    def groundings(self) -> tuple[GroundValue, ...]:
        model = self.model

        observ_fluents = model.observ_fluents  # type: ignore
        non_fluents = model.non_fluents  # type: ignore

        non_fluent_groundings = get_groundings(model, non_fluents)  # type: ignore
        observ_groundings = get_groundings(model, observ_fluents)  # type: ignore

        all_groundings = observ_groundings | non_fluent_groundings

        all_groundings = [rddl_ground_to_tuple(g) for g in all_groundings]

        return tuple(sorted(all_groundings))
