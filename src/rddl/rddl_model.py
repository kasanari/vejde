from itertools import chain
from pyRDDLGym.core.compiler.model import RDDLLiftedModel  # type: ignore
from functools import cache, cached_property
from copy import copy
from model.base_model import BaseModel
from .utils import get_groundings


class RDDLModel(BaseModel):
    def __init__(self, model: RDDLLiftedModel) -> None:
        self.model = model

    @cache
    def arity(self, fluent: str) -> int:
        return self.arities[fluent]

    @cache
    def fluents_of_arity(self, arity: int) -> tuple[str, ...]:
        return self._fluents_of_arity[arity]

    @cache
    def idx_to_fluent(self, idx: int) -> str:
        return self.fluents[idx]

    @cache
    def idx_to_type(self, idx: int) -> str:
        return self._idx_to_type[idx]

    @cache
    def fluent_to_idx(self, relation: str) -> int:
        return self._rel_to_idx[relation]

    @cache
    def fluent_params(self, variable: str) -> tuple[str, ...]:
        return self._variable_params[variable]

    @cache
    def fluent_param(self, fluent: str, position: int) -> str:
        """Types/class of the variable/object the fluent/predicate takes as parameter in a given position. Can be seen as the column name in a database table."""
        return self._variable_params[fluent][position]

    @cache
    def fluent_range(self, fluent: str) -> type:
        return self.variable_ranges[fluent]

    @cache
    def idx_to_action(self, idx: int) -> str:
        return self.action_fluents[idx]

    @cache
    def action_to_idx(self, action: str) -> int:
        return self.action_fluents.index(action)

    @cached_property
    def fluents(self) -> tuple[str, ...]:
        x = sorted(
            list(
                chain(
                    self.model.state_fluents.keys(),
                    self.model.non_fluents.keys(),
                    self.model.observ_fluents.keys(),
                    self.model.action_fluents.keys(),
                )
            )
        )

        return tuple(["None"] + x)

    @cached_property
    def types(self) -> tuple[str, ...]:
        return tuple(self._idx_to_type)

    @cached_property
    def _idx_to_type(self) -> list[str]:
        return ["None"] + sorted(set(self._obj_to_type.values()))

    @cached_property
    def _obj_to_type(self) -> dict[str, str]:
        model: RDDLLiftedModel = self.model  # type: ignore
        object_to_type: dict[str, str] = copy(model.object_to_type)  # type: ignore
        return object_to_type

    @cached_property
    def num_types(self) -> int:
        return len(self._idx_to_type)

    @cached_property
    def variable_ranges(self) -> dict[str, type]:
        mapping = {
            "bool": bool,
            "int": int,
            "real": float,
        }

        # mapping.update({e: e for e in self.model.enum_types})

        variable_ranges: dict[str, type] = {
            key: mapping[value]
            for key, value in self.model._variable_ranges.items()  # type: ignore
        }

        variable_ranges["None"] = bool

        return variable_ranges

    @cached_property
    def _variable_params(self) -> dict[str, tuple[str, ...]]:
        variable_params: dict[str, list[str]] = copy(self.model.variable_params)  # type: ignore
        variable_params["None"] = []
        return {k: tuple(v) for k, v in variable_params.items()}

    @cached_property
    def _fluents_of_arity(self) -> dict[int, tuple[str, ...]]:
        arities: dict[str, int] = self.arities
        return {
            value: tuple([k for k, v in arities.items() if v == value])
            for _, value in arities.items()
        }

    @cached_property
    def groundings(self) -> list[str]:
        model = self.model

        state_fluents = model.state_fluents  # type: ignore
        non_fluents = model.non_fluents  # type: ignore

        non_fluent_groundings = get_groundings(model, non_fluents)  # type: ignore
        state_groundings = get_groundings(model, state_fluents)  # type: ignore

        all_groundings = state_groundings | non_fluent_groundings

        return sorted(all_groundings)

    @cached_property
    def action_fluents(self) -> list[str]:
        model = self.model
        action_fluents = model.action_fluents  # type: ignore
        return ["None"] + sorted(action_fluents.keys())  # type: ignore

    @cached_property
    def num_actions(self) -> int:
        return len(self.action_fluents)

    @cached_property
    def action_groundings(self) -> set[str]:
        return get_groundings(self.model, self.model.action_fluents) | {"None"}  # type: ignore

    @cached_property
    def num_fluents(self) -> int:
        return len(self.fluents)

    @cache
    def type_to_idx(self, type: str) -> int:
        return self._type_to_idx[type]

    @cached_property
    def _type_to_idx(self) -> dict[str, int]:
        return {
            symb: idx for idx, symb in enumerate(self._idx_to_type)
        }  # 0 is reserved for padding

    @cached_property
    def _rel_to_idx(self) -> dict[str, int]:
        return {
            symb: idx for idx, symb in enumerate(self.fluents)
        }  # 0 is reserved for padding

    @cached_property
    def arities(self) -> dict[str, int]:
        return {key: len(value) for key, value in self._variable_params.items()}
