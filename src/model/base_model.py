from functools import cache
from abc import ABC, abstractmethod


class BaseModel(ABC):
    @property
    @abstractmethod
    @cache
    def num_types(self) -> int:
        """The number of object types/classes in the model."""
        ...

    @property
    @abstractmethod
    @cache
    def num_actions(self) -> int:
        """The number of action predicates/fluents in the model."""
        ...

    @abstractmethod
    @cache
    def variable_range(self, fluent: str) -> type:
        """The type of the fluent/predicate."""
        ...

    @abstractmethod
    @cache
    def fluent_params(self, fluent: str) -> list[str]:
        """Types/classes of the variables/objects the fluent/predicate takes as parameters. Can be seen as the column names in a database table."""
        ...

    @abstractmethod
    @cache
    def fluent_param(self, fluent: str, position: int) -> str:
        """Types/class of the variable/object the fluent/predicate takes as parameter in a given position. Can be seen as the column name in a database table."""
        ...

    @abstractmethod
    @cache
    def type_attributes(self, type: str) -> list[str]:
        """unary attributes/predicates/fluents for object type."""
        ...

    @abstractmethod
    @cache
    def fluents_of_arity(self, arity: int) -> list[str]:
        """fluents/predicates of a given arity (number of parameters/variables/objects)."""
        ...

    @property
    @abstractmethod
    @cache
    def groundings(self) -> list[str]:
        """
        A list of all possible grounded variables in the language.
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
        """groundings of action fluents/variables.
        one the form: relation___object1__object2__...__objectN
        """
        ...

    @property
    @abstractmethod
    @cache
    def num_relations(self) -> int:
        """The number of relations/predicates in the model. This includes nullary and unary relations, which may also be called constants and attributes."""
        ...

    @abstractmethod
    @cache
    def type_to_idx(self, type: str) -> int:
        """
        A mapping from object type to an index.
        This should be consistent across all instances of the same domain.
        Note that 0 is reserved for padding.
        """
        ...

    @abstractmethod
    @cache
    def idx_to_type(self, idx: int) -> str:
        """
        A mapping from an index to an object type.
        This should be consistent across all instances of the same domain.
        Note that 0 is reserved for padding.
        """
        ...

    @abstractmethod
    @cache
    def rel_to_idx(self, relation: str) -> int:
        """
        A mapping from a relation/predicate to an index.
        This should be consistent across all instances of the same domain.
        Note that 0 is reserved for padding.
        """
        ...

    @abstractmethod
    @cache
    def idx_to_relation(self, idx: int) -> str:
        """
        A mapping from an index to a relation/predicate.
        This should be consistent across all instances of the same domain.
        Note that 0 is reserved for padding.
        """

    ...

    @abstractmethod
    @cache
    def idx_to_action(self, idx: int) -> str:
        """
        A mapping from an index to an action fluent.
        This should be consistent across all instances of the same domain.
        Note that 0 is reserved for padding/default actions.
        """
        ...

    @abstractmethod
    @cache
    def arity(self, fluent: str) -> int:
        "The arity (number of variables) of each relation/predicate in the model."
        ...
