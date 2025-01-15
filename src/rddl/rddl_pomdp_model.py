from rddl.rddl_model import RDDLModel
from .utils import get_groundings
from functools import cache, cached_property


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

    @cached_property
    def fluents(self) -> tuple[str, ...]:
        x = list(self.model.non_fluents.keys())
        x = x + list(self.model.observ_fluents.keys())
        x = x + list(self.model.action_fluents.keys())
        x = x + list(self.model.state_fluents.keys())

        return tuple(["None"] + sorted(x))
