from itertools import chain
from pyRDDLGym.core.compiler.model import RDDLLiftedModel  # type: ignore
from functools import cache, cached_property
from copy import copy
from regawa.model.base_model import BaseModel
from .rddl_utils import get_groundings, rddl_ground_to_tuple


class RDDLModel(BaseModel):
    def __init__(self, model: RDDLLiftedModel) -> None:
        self.model = model

    @cached_property
    def enum_fluents(self) -> dict[str, list[str]]:
        fluents = list(
            chain(
                *(
                    self.model.state_fluents.keys(),
                    self.model.non_fluents.keys(),
                    self.model.observ_fluents.keys(),
                    self.model.action_fluents.keys(),
                )
            )
        )

        enum_types = self.model.enum_types

        enum_fluents = {
            r: [f for f in fluents if r == self.model.variable_ranges[f]]
            for r in enum_types
        }

        return enum_fluents

    @cache
    def _enum_values(self, enum: str):
        object_to_type = self.model.object_to_type
        return [o for o, t in object_to_type.items() if t == enum]

    @cached_property
    def enum_values(self) -> dict[str, list[str]]:
        return {t: self._enum_values(t) for t in self.model.enum_types}

    @cached_property
    def combined_enum_fluents(self) -> dict[str, str]:
        enum_fluents = self.enum_fluents
        enum_values = self.enum_values

        combined_enum_fluents = {
            f"{f}^^^{i}": f
            for enum, values in enum_values.items()
            for f in enum_fluents[enum]
            for i, v in enumerate(values)
        }

        return combined_enum_fluents

    def combined_enum_fluent(self, fluent: str):
        enum = self.model.variable_ranges[fluent]
        enum_values = self.enum_values[enum]

        return [f"{fluent}^^^{i}" for i, _ in enumerate(enum_values)]

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
        x: chain[str] = chain(
            self.model.state_fluents.keys(),
            self.model.non_fluents.keys(),
            self.model.observ_fluents.keys(),
            self.model.action_fluents.keys(),
        )

        fluents_with_enums = list(chain(*self.enum_fluents.values()))

        y = filter(lambda e: e not in fluents_with_enums, x)

        combined_enum_fluents = self.combined_enum_fluents.keys()

        x = chain(y, combined_enum_fluents)

        return tuple(["None"] + sorted(x))

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

        mapping.update({e: bool for e in self.model.enum_types})

        variable_ranges: dict[str, type] = {
            key: mapping[value]
            for key, value in self.model._variable_ranges.items()  # type: ignore
        }

        variable_ranges["None"] = bool

        combined_enum_fluents = {f: bool for f in self.combined_enum_fluents}

        variable_ranges = {
            f: variable_ranges.get(f, combined_enum_fluents.get(f))
            for f in self.fluents
        }

        return variable_ranges

    @cached_property
    def _variable_params(self) -> dict[str, tuple[str, ...]]:
        variable_params: dict[str, list[str]] = copy(self.model.variable_params)  # type: ignore
        variable_params["None"] = []

        combined_enum_fluent_params = {
            f: variable_params[v] for f, v in self.combined_enum_fluents.items()
        }

        variable_params = {
            f: variable_params.get(f, combined_enum_fluent_params.get(f))
            for f in self.fluents
        }

        return {k: tuple(v) for k, v in variable_params.items()}

    @cached_property
    def _fluents_of_arity(self) -> dict[int, tuple[str, ...]]:
        arities: dict[str, int] = self.arities
        return {
            value: tuple([k for k, v in arities.items() if v == value])
            for _, value in arities.items()
        }

    @cached_property
    def groundings(self) -> list[tuple[str, ...]]:
        model = self.model

        state_fluents = model.state_fluents  # type: ignore
        non_fluents = model.non_fluents  # type: ignore

        non_fluent_groundings = get_groundings(model, non_fluents)  # type: ignore
        state_groundings = get_groundings(model, state_fluents)  # type: ignore

        all_groundings = state_groundings | non_fluent_groundings

        all_groundings = [rddl_ground_to_tuple(g) for g in all_groundings]

        return sorted(all_groundings)

    @cached_property
    def action_fluents(self) -> list[str]:
        model = self.model
        action_fluents = model.action_fluents  # type: ignore

        enum_fluents = list(chain(*self.enum_fluents.values()))

        new_fluents = chain(
            *[self.combined_enum_fluent(f) for f in action_fluents if f in enum_fluents]
        )

        action_fluents = {f for f in action_fluents if f not in enum_fluents}
        action_fluents = action_fluents | set(new_fluents)

        return ["None"] + sorted(action_fluents)  # type: ignore

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
