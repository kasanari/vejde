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

    @abstractmethod
    @cache
    def variable_range(self, fluent: str) -> type:
        """The type of the fluent."""
        ...

    @abstractmethod
    @cache
    def variable_params(self, fluent: str) -> list[str]:
        """Types of the variables the fluent takes."""
        ...

    @abstractmethod
    @cache
    def type_attributes(self, type: str) -> list[str]:
        """unary attributes for object type."""
        ...

    @abstractmethod
    @cache
    def fluents_of_arity(self, arity: int) -> list[str]:
        """fluents of a given arity."""
        ...

    @property
    @abstractmethod
    @cache
    def groundings(self) -> list[str]:
        """
        A list of all possible grounded variables in the model.
        On the form:
        relation___object1__object2__...__objectN
        """
        ...

    @property
    @abstractmethod
    @cache
    def action_fluents(self) -> list[str]:
        """relations/fluents/predicates that can be used as actions in the model."""
        ...

    @property
    @abstractmethod
    @cache
    def action_groundings(self) -> set[str]:
        """groundings of action fluents."""
        ...

    @property
    @abstractmethod
    @cache
    def num_relations(self) -> int:
        "The number of relations in the model. This includes nullary and unary relations, which may also be called constants and attributes."
        ...

    @property
    @abstractmethod
    @cache
    def num_objects(self) -> int:
        "The number of objects in the instance."
        ...

    @abstractmethod
    @cache
    def obj_to_type(self, obj: str) -> str:
        "The type/class of an object instance/variable."
        ...

    @abstractmethod
    @cache
    def type_to_idx(self, type: str) -> int:
        "A mapping from object type to an index."
        ...

    @abstractmethod
    @cache
    def idx_to_type(self, idx: int) -> str: ...

    @abstractmethod
    @cache
    def rel_to_idx(self, relation: str) -> int: ...

    @abstractmethod
    @cache
    def idx_to_relation(self, idx: int) -> str: ...

    @abstractmethod
    @cache
    def obj_to_idx(self, obj: str) -> int: ...

    @abstractmethod
    @cache
    def idx_to_object(self, idx: int) -> str: ...

    @abstractmethod
    def idx_to_action(self, idx: int) -> str: ...

    @abstractmethod
    @cache
    def arity(self, fluent: str) -> int:
        "The arity (number of variables) of each relation/predicate in the model."
        ...
