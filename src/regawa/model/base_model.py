from abc import ABC, abstractmethod
from functools import cache, cached_property


class BaseModel(ABC):
    @cached_property
    @abstractmethod
    def num_types(self) -> int:
        """The number of object types/classes in the model."""
        ...

    @cached_property
    @abstractmethod
    def num_actions(self) -> int:
        """The number of action predicates/fluents in the model."""
        ...

    @abstractmethod
    @cache
    def fluent_range(self, fluent: str) -> type:
        """The type of the fluent/predicate."""
        ...

    @abstractmethod
    @cache
    def fluent_params(self, fluent: str) -> tuple[str, ...]:
        """Types/classes of the variables/objects the fluent/predicate takes as parameters. Can be seen as the column names in a database table."""
        ...

    @abstractmethod
    @cache
    def fluent_param(self, fluent: str, position: int) -> str:
        """Types/class of the variable/object the fluent/predicate takes as parameter in a given position. Can be seen as the column name in a database table."""
        ...

    @cached_property
    @abstractmethod
    def action_fluents(self) -> tuple[str, ...]:
        """relations/fluents/predicates that can be used as actions in the model."""
        ...

    @cached_property
    @abstractmethod
    def num_fluents(self) -> int:
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
    def fluent_to_idx(self, relation: str) -> int:
        """
        A mapping from a relation/predicate to an index.
        This should be consistent across all instances of the same domain.
        Note that 0 is reserved for padding.
        """
        ...

    @cached_property
    @abstractmethod
    def fluents(self) -> tuple[str, ...]:
        """A list of all relations/predicates in the model."""
        ...

    @cached_property
    @abstractmethod
    def types(self) -> tuple[str, ...]:
        """A list of all object types/classes in the model."""
        ...

    @abstractmethod
    @cache
    def idx_to_fluent(self, idx: int) -> str:
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
    def action_to_idx(self, action: str) -> int:
        """
        A mapping from an action fluent to an index.
        This should be consistent across all instances of the same domain.
        Note that 0 is reserved for padding/default actions.
        """
        ...

    @abstractmethod
    @cache
    def arity(self, fluent: str) -> int:
        "The arity (number of variables) of each relation/predicate in the model."
        ...
