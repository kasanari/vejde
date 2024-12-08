from pyRDDLGym.core.compiler.model import RDDLLiftedModel, RDDLPlanningModel  # type: ignore
from functools import cache
from copy import copy
from .base_model import BaseModel
from wrappers.utils import predicate
from wrappers.utils import get_groundings


def skip_fluent(key: str, variable_ranges: dict[str, str]) -> bool:
    return variable_ranges[predicate(key)] != "bool" or key == "noop"


class RDDLModel(BaseModel):
    def __init__(self, model: RDDLLiftedModel) -> None:
        self.model = model

    @property
    @cache
    def idx_to_type(self) -> list[str]:
        return sorted(set(self.obj_to_type.values()))

    @property
    @cache
    def obj_to_type(self) -> dict[str, str]:
        model: RDDLLiftedModel = self.model  # type: ignore
        object_to_type: dict[str, str] = copy(model.object_to_type)  # type: ignore
        return object_to_type

    @property
    @cache
    def num_types(self) -> int:
        return len(self.idx_to_type)

    @property
    @cache
    def non_fluent_values(self) -> dict[str, int]:
        # model = self.model
        # return dict(
        #     model.ground_vars_with_values(model.non_fluents)  # type: ignore
        # )

        nf_vals = {}

        non_fluents = self.model.ast.non_fluents.init_non_fluent  # type: ignore
        for (name, params), value in non_fluents:
            gname = RDDLPlanningModel.ground_var(name, params)
            nf_vals[gname] = value

        return nf_vals

    @property
    @cache
    def num_edges(self) -> int:
        return sum(self.arities[predicate(g)] for g in self.groundings)

    @property
    @cache
    def variable_ranges(self) -> dict[str, str]:
        variable_ranges: dict[str, str] = self.model._variable_ranges  # type: ignore
        variable_ranges["noop"] = "bool"
        return variable_ranges

    @property
    @cache
    def variable_params(self) -> dict[str, list[str]]:
        variable_params: dict[str, list[str]] = copy(self.model.variable_params)  # type: ignore
        variable_params["noop"] = []
        return variable_params

    @property
    @cache
    def type_to_arity(self) -> dict[str, int]:
        vp = self.model.variable_params  # type: ignore
        return {
            value[0]: [k for k, v in vp.items() if v == value]  # type: ignore
            for _, value in vp.items()  # type: ignore
            if len(value) == 1  # type: ignore
        }

    @property
    @cache
    def arities_to_fluent(self) -> dict[int, list[str]]:
        arities: dict[str, int] = self.arities
        return {
            value: [k for k, v in arities.items() if v == value]
            for _, value in arities.items()
        }

    @property
    @cache
    def idx_to_object(self) -> list[str]:
        object_terms: list[str] = list(self.model.object_to_index.keys())  # type: ignore
        object_list = sorted(object_terms)
        return object_list

    @property
    @cache
    def idx_to_relation(self) -> list[str]:
        relation_list = sorted(set(predicate(g) for g in self.groundings))
        return relation_list

    @property
    @cache
    def all_groundings(self) -> list[str]:
        model = self.model

        state_fluents = model.state_fluents  # type: ignore

        non_fluent_groundings = set(self.non_fluent_values.keys())
        state_groundings: set[str] = get_groundings(model, state_fluents)  # type: ignore

        g = state_groundings | non_fluent_groundings

        return sorted(g)

    @property
    @cache
    def action_fluents(self) -> list[str]:
        model = self.model
        action_fluents = model.action_fluents  # type: ignore
        return ["noop"] + sorted(action_fluents)  # type: ignore

    @property
    @cache
    def action_groundings(self) -> set[str]:
        return get_groundings(self.model, self.model.action_fluents) | {"noop"}  # type: ignore

    @property
    @cache
    def num_relations(self) -> int:
        return len(self.idx_to_relation)

    @property
    @cache
    def num_objects(self) -> int:
        return len(self.idx_to_object)

    @property
    @cache
    def type_to_idx(self) -> dict[str, int]:
        return {
            symb: idx + 1 for idx, symb in enumerate(self.idx_to_type)
        }  # 0 is reserved for padding

    @property
    @cache
    def rel_to_idx(self) -> dict[str, int]:
        return {
            symb: idx + 1 for idx, symb in enumerate(self.idx_to_relation)
        }  # 0 is reserved for padding

    @property
    @cache
    def obj_to_idx(self) -> dict[str, int]:
        return {
            symb: idx + 1 for idx, symb in enumerate(self.idx_to_object)
        }  # 0 is reserved for padding

    @property
    @cache
    def arities(self) -> dict[str, int]:
        return {key: len(value) for key, value in self.variable_params.items()}

    @property
    @cache
    def groundings(self):
        return sorted(
            [g for g in self.all_groundings if not skip_fluent(g, self.variable_ranges)]
        )
