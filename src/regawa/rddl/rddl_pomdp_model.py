from .rddl_model import RDDLModel
from .rddl_utils import get_groundings
from functools import cache


class RDDLPOMDPModel(RDDLModel):
    @property
    @cache
    def groundings(self) -> list[str]:
        model = self.model

        observ_fluents = model.observ_fluents  # type: ignore
        non_fluents = model.non_fluents  # type: ignore

        non_fluent_groundings = get_groundings(model, non_fluents)  # type: ignore
        observ_groundings = get_groundings(model, observ_fluents)  # type: ignore

        all_groundings = observ_groundings | non_fluent_groundings

        return sorted(all_groundings)
