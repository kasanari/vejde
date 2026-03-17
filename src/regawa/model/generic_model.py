from __future__ import annotations

from collections.abc import Sequence
from functools import cache, cached_property

from .base_model import BaseModel
from .model_checker import check_model


class GenericModel(BaseModel):
    """A generic implementation of the BaseModel interface. Mainly intended for deserialization purposes."""

    def __init__(
        self,
        fluents: Sequence[str],
        types: Sequence[str],
        action_fluents: Sequence[str],
        fluent_params: dict[str, Sequence[str]],
        fluent_ranges: dict[str, type],
    ):
        self._fluents = fluents
        self._action_fluents = action_fluents
        self._fluent_params = fluent_params
        self._fluent_ranges = fluent_ranges
        self._types = types

    @cache
    def arity(self, fluent: str) -> int:
        return self.arities[fluent]

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
        return tuple(self._fluent_params[variable])

    @cache
    def fluent_param(self, fluent: str, position: int) -> str:
        """Types/class of the variable/object the fluent/predicate takes as parameter in a given position. Can be seen as the column name in a database table."""
        return self._fluent_params[fluent][position]

    @cache
    def fluent_range(self, fluent: str) -> type:
        return self._fluent_ranges[fluent]

    @cache
    def idx_to_action(self, idx: int) -> str:
        return self.action_fluents[idx]

    @cache
    def action_to_idx(self, action: str) -> int:
        return self.action_fluents.index(action)

    @cache
    def type_to_idx(self, _type: str) -> int:
        return self._type_to_idx[_type]

    @cached_property
    def fluents(self) -> tuple[str, ...]:
        return tuple(self._fluents)

    @cached_property
    def types(self) -> tuple[str, ...]:
        return tuple(self._idx_to_type)

    @cached_property
    def _idx_to_type(self) -> Sequence[str]:
        return self._types

    @cached_property
    def num_types(self) -> int:
        return len(self._idx_to_type)

    @cached_property
    def action_fluents(self) -> tuple[str, ...]:
        return tuple(self._action_fluents)

    @cached_property
    def num_actions(self) -> int:
        return len(self.action_fluents)

    @cached_property
    def num_fluents(self) -> int:
        return len(self.fluents)

    @cached_property
    def _type_to_idx(self) -> dict[str, int]:
        return {symb: idx for idx, symb in enumerate(self._idx_to_type)}

    @cached_property
    def _rel_to_idx(self) -> dict[str, int]:
        return {symb: idx for idx, symb in enumerate(self.fluents)}

    @cached_property
    def arities(self) -> dict[str, int]:
        return {key: len(value) for key, value in self._fluent_params.items()}

    @classmethod
    def from_json(cls, json_str: str) -> GenericModel:
        import json

        data = json.loads(json_str)

        string_to_type = {
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
        }

        model = cls(
            fluents=data["fluents"],
            types=data["types"],
            action_fluents=data["action_fluents"],
            fluent_params={k: tuple(v) for k, v in data["fluent_params"].items()},
            fluent_ranges={
                k: string_to_type[v] for k, v in data["fluent_ranges"].items()
            },
        )
        try:
            check_model(model)
        except Exception as e:
            raise ValueError(f"Deserialized model does not pass check: {e}") from e
        return model
