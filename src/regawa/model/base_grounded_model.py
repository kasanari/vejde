from abc import ABC, abstractmethod
from functools import cache, cached_property
from typing import Any

GroundValue = tuple[str, ...]


class BaseGroundedModel(ABC):
    "Grounded model base class. This is primarily for special usecases when instance-specific information is needed, and is not required to use Vejde."

    @cached_property
    @abstractmethod
    def groundings(self) -> tuple[GroundValue, ...]:
        """
        A list of all possible grounded facts based on the problem instance.
        on the form: (relation, object1, object2,..., objectN)
        """
        ...

    @cached_property
    @abstractmethod
    def action_groundings(self) -> tuple[GroundValue, ...]:
        """groundings of action fluents/variables.
        on the form: (relation, object1, object2,..., objectN)
        """
        ...

    @cached_property
    @abstractmethod
    def constant_groundings(self) -> tuple[GroundValue, ...]:
        """Groundings assumed to be constant in the model."""
        ...

    @cache
    @abstractmethod
    def constant_value(self, constant_grounding: GroundValue) -> Any:
        """The constant value of a constant grounding."""
        ...
