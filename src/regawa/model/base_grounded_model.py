from abc import ABC, abstractmethod
from functools import cache, cached_property
from typing import Any

from regawa.model import GroundValue


class BaseGroundedModel(ABC):
    @cached_property
    @abstractmethod
    def groundings(self) -> tuple[GroundValue, ...]:
        """
        A list of all possible grounded variables in the language.
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
