from typing import Any
from pyRDDLGym.core.compiler.model import RDDLLiftedModel  # type: ignore
from functools import cache, cached_property
from regawa.model.base_grounded_model import BaseGroundedModel
from .rddl_utils import get_groundings, rddl_ground_to_tuple
from regawa import GroundValue
from pyRDDLGym.core.compiler.model import RDDLPlanningModel  # type: ignore


class RDDLGroundedModel(BaseGroundedModel):
    def __init__(self, model: RDDLLiftedModel) -> None:
        self.model = model

    @cached_property
    def groundings(self) -> tuple[GroundValue, ...]:
        model = self.model

        state_fluents = model.state_fluents  # type: ignore
        non_fluents = model.non_fluents  # type: ignore

        non_fluent_groundings = get_groundings(model, non_fluents)  # type: ignore
        state_groundings = get_groundings(model, state_fluents)  # type: ignore

        all_groundings = state_groundings | non_fluent_groundings

        all_groundings = [rddl_ground_to_tuple(g) for g in all_groundings]

        return tuple(sorted(all_groundings))

    @cached_property
    def action_groundings(self) -> tuple[GroundValue, ...]:
        return get_groundings(self.model, self.model.action_fluents) | {"None"}  # type: ignore

    @cached_property
    def constant_groundings(self) -> tuple[GroundValue, ...]:
        return tuple(self._non_fluent_vals.keys())

    @cache
    def constant_value(self, constant_grounding: GroundValue) -> Any:
        return self._non_fluent_vals[constant_grounding]

    @cached_property
    def _non_fluents(self) -> list[tuple[str, Any]]:
        return (
            self.model.ast.non_fluents.init_non_fluent  # type: ignore
            if hasattr(self.model.ast.non_fluents, "init_non_fluent")  # type: ignore
            else []
        )

    @cached_property
    def _non_fluent_vals(self) -> dict[GroundValue, Any]:
        return {
            rddl_ground_to_tuple(RDDLPlanningModel.ground_var(name, params)): value
            for (name, params), value in self._non_fluents
        }
