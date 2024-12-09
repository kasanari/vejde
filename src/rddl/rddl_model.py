from collections.abc import Callable
from pyRDDLGym.core.compiler.model import RDDLLiftedModel  # type: ignore
from functools import cache
from copy import copy
from ..model.base_model import BaseModel
from wrappers.utils import predicate
from .utils import get_groundings


class RDDLModel(BaseModel):
    def __init__(self, model: RDDLLiftedModel) -> None:
        self.model = model

    @cache
    def arity(self, fluent: str) -> int:
        return self.arities[fluent]

    @cache
    def fluents_of_arity(self, arity: int) -> list[str]:
        return self._fluents_of_arity[arity]

    @cache
    def idx_to_object(self, idx: int) -> str:
        return self._idx_to_object[idx]

    @cache
    def idx_to_relation(self, idx: int) -> str:
        return self._idx_to_relation[idx]

    @cache
    def idx_to_type(self, idx: int) -> str:
        return self._idx_to_type[idx]

    @cache
    def obj_to_type(self, obj: str) -> str:
        return self._obj_to_type[obj]

    @cache
    def rel_to_idx(self, relation: str) -> int:
        return self._rel_to_idx[relation]

    @cache
    def type_attributes(self, type: str) -> list[str]:
        return self._type_attributes[type]

    @cache
    def variable_params(self, variable: str) -> list[str]:
        return self._variable_params[variable]

    @cache
    def variable_range(self, fluent: str) -> type:
        return self.variable_ranges[fluent]

    @cache
    def idx_to_action(self, idx: int) -> str:
        return self.action_fluents[idx]

    @property
    @cache
    def _idx_to_type(self) -> list[str]:
        return sorted(set(self._obj_to_type.values()))

    @property
    @cache
    def _obj_to_type(self) -> dict[str, str]:
        model: RDDLLiftedModel = self.model  # type: ignore
        object_to_type: dict[str, str] = copy(model.object_to_type)  # type: ignore
        return object_to_type

    @property
    @cache
    def num_types(self) -> int:
        return len(self._idx_to_type)

    @property
    @cache
    def variable_ranges(self) -> dict[str, type]:
        mapping = {
            "bool": bool,
            "int": int,
            "real": float,
        }

        variable_ranges: dict[str, type] = {
            key: mapping[value]
            for key, value in self.model._variable_ranges.items()  # type: ignore
        }
        variable_ranges["noop"] = bool

        return variable_ranges

    @property
    @cache
    def _variable_params(self) -> dict[str, list[str]]:
        variable_params: dict[str, list[str]] = copy(self.model.variable_params)  # type: ignore
        variable_params["noop"] = []
        return variable_params

    @property
    @cache
    def _type_attributes(self) -> dict[str, list[str]]:
        vp = self.model.variable_params  # type: ignore
        return {
            value[0]: [k for k, v in vp.items() if v == value]  # type: ignore
            for _, value in vp.items()  # type: ignore
            if len(value) == 1  # type: ignore
        }

    @property
    @cache
    def _fluents_of_arity(self) -> dict[int, list[str]]:
        arities: dict[str, int] = self.arities
        return {
            value: [k for k, v in arities.items() if v == value]
            for _, value in arities.items()
        }

    @property
    @cache
    def _idx_to_object(self) -> list[str]:
        object_terms: list[str] = list(self.model.object_to_index.keys())  # type: ignore
        object_list = sorted(object_terms)
        return object_list

    @property
    @cache
    def _idx_to_relation(self) -> list[str]:
        relation_list = sorted(set(predicate(g) for g in self.groundings))
        return relation_list

    @property
    @cache
    def groundings(self) -> list[str]:
        model = self.model

        state_fluents = model.state_fluents  # type: ignore
        non_fluents = model.non_fluents  # type: ignore

        non_fluent_groundings = get_groundings(model, non_fluents)  # type: ignore
        state_groundings = get_groundings(model, state_fluents)  # type: ignore

        all_groundings = state_groundings | non_fluent_groundings

        return sorted(all_groundings)

    @property
    @cache
    def action_fluents(self) -> list[str]:
        model = self.model
        action_fluents = model.action_fluents  # type: ignore
        return ["noop"] + sorted(action_fluents)  # type: ignore

    @property
    @cache
    def num_actions(self) -> int:
        return len(self.action_fluents)

    @property
    @cache
    def action_groundings(self) -> set[str]:
        return get_groundings(self.model, self.model.action_fluents) | {"noop"}  # type: ignore

    @property
    @cache
    def num_relations(self) -> int:
        return len(self._idx_to_relation)

    @property
    @cache
    def num_objects(self) -> int:
        return len(self._idx_to_object)

    @cache
    def type_to_idx(self, type: str) -> int:
        return self._type_to_idx[type]

    @property
    @cache
    def _type_to_idx(self) -> dict[str, int]:
        return {
            symb: idx + 1 for idx, symb in enumerate(self._idx_to_type)
        }  # 0 is reserved for padding

    @property
    @cache
    def _rel_to_idx(self) -> dict[str, int]:
        return {
            symb: idx + 1 for idx, symb in enumerate(self._idx_to_relation)
        }  # 0 is reserved for padding

    @cache
    def obj_to_idx(self, obj: str) -> int:
        return self._obj_to_idx[obj]

    @property
    @cache
    def _obj_to_idx(self) -> dict[str, int]:
        return {
            symb: idx + 1 for idx, symb in enumerate(self._idx_to_object)
        }  # 0 is reserved for padding

    @property
    @cache
    def arities(self) -> dict[str, int]:
        return {key: len(value) for key, value in self._variable_params.items()}
