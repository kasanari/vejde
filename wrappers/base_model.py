from functools import cache
from abc import ABC, abstractmethod


class BaseModel(ABC):
    @property
    @abstractmethod
    @cache
    def num_types(self) -> int: ...

    @property
    @abstractmethod
    @cache
    def num_actions(self) -> int: ...

    @property
    @abstractmethod
    @cache
    def num_edges(self) -> int:
        """
        This can be calculated as the sum of the arities of all groundings.
        """
        ...

    @property
    @abstractmethod
    @cache
    def variable_ranges(self) -> dict[str, str]: ...

    @property
    @abstractmethod
    @cache
    def variable_params(self) -> dict[str, list[str]]: ...

    @property
    @abstractmethod
    @cache
    def type_to_arity(self) -> dict[str, int]: ...

    @property
    @abstractmethod
    @cache
    def arities_to_fluent(self) -> dict[int, list[str]]: ...

    @property
    @abstractmethod
    @cache
    def groundings(self) -> list[str]:
        """
        A list of all grounded variables in the model.
        On the form:
        relation___object1__object2__...__objectN
        """
        ...

    @property
    @abstractmethod
    @cache
    def action_fluents(self) -> list[str]: ...

    @property
    @abstractmethod
    @cache
    def action_groundings(self) -> set[str]: ...

    @property
    @abstractmethod
    @cache
    def num_relations(self) -> int: ...

    @property
    @abstractmethod
    @cache
    def non_fluent_values(self) -> dict[str, int]:
        """
        A dictionary of grounded variables and their values in the instance model assumed to be constant over time.
        """
        ...

    @property
    @abstractmethod
    @cache
    def num_objects(self) -> int: ...

    @property
    @abstractmethod
    @cache
    def obj_to_type(self) -> dict[str, str]: ...

    @property
    @abstractmethod
    @cache
    def type_to_idx(self) -> dict[str, int]: ...

    @property
    @abstractmethod
    @cache
    def idx_to_type(self) -> list[str]: ...

    @property
    @abstractmethod
    @cache
    def rel_to_idx(self) -> dict[str, int]: ...

    @property
    @abstractmethod
    @cache
    def idx_to_relation(self) -> list[str]: ...

    @property
    @abstractmethod
    @cache
    def obj_to_idx(self) -> dict[str, int]: ...

    @property
    @abstractmethod
    @cache
    def idx_to_object(self) -> list[str]: ...

    @property
    @abstractmethod
    @cache
    def arities(self) -> dict[str, int]: ...
