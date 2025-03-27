from functools import cache, cached_property
from typing import Any

import numpy as np
from pyRDDLGym.core.compiler.model import RDDLLiftedModel  # type: ignore
from pyRDDLGym.core.compiler.model import RDDLPlanningModel  # type: ignore

from regawa.model import GroundValue
from regawa.model.base_grounded_model import BaseGroundedModel

from .rddl_utils import get_groundings, rddl_ground_to_tuple


class RDDLGroundedModel(BaseGroundedModel):
    def __init__(
        self,
        model: RDDLLiftedModel,
        all_non_fluents: bool = True,
        remove_false: bool = False,
    ) -> None:
        self.model = model
        self.all_non_fluents = all_non_fluents
        self.remove_false = remove_false

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
        return (
            tuple(self._all_non_fluent_vals.keys())
            if self.all_non_fluents
            else tuple(self._non_fluent_vals.keys())
        )

    @cache
    def constant_value(self, constant_grounding: GroundValue) -> Any:
        return (
            self._all_non_fluent_vals[constant_grounding]
            if self.all_non_fluents
            else self._non_fluent_vals[constant_grounding]
        )

    @cached_property
    def _non_fluents(self) -> list[tuple[str, Any]]:
        """Non-fluents that are "observed" in the instance file."""
        return (
            self.model.ast.non_fluents.init_non_fluent  # type: ignore
            if hasattr(self.model.ast.non_fluents, "init_non_fluent")  # type: ignore
            else []
        )

    @cached_property
    def _all_non_fluent_vals(self) -> dict[GroundValue, Any]:
        """All non-fluents, including those that are not observed in the instance file."""
        return (
            {
                rddl_ground_to_tuple(g): v
                for g, v in self.model.ground_vars_with_values(
                    self.model.non_fluents
                ).items()
            }
            if not self.remove_false
            else {
                rddl_ground_to_tuple(g): v
                for g, v in self.model.ground_vars_with_values(
                    self.model.non_fluents
                ).items()
                if v != np.bool_(False)
            }
        )

    @cached_property
    def _non_fluent_vals(self) -> dict[GroundValue, Any]:
        return {
            rddl_ground_to_tuple(RDDLPlanningModel.ground_var(name, params)): value
            for (name, params), value in self._non_fluents
        }
