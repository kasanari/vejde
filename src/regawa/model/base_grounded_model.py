from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import cache, cached_property
import numpy as np

type Grounding = tuple[str, ...]
type GroundingValueType = bool | int | float | np.bool_
type ObservableGroundingValueType = GroundingValueType | None
GroundObs = dict[Grounding, GroundingValueType]
type ObservableGroundObs = Mapping[Grounding, ObservableGroundingValueType]

StackedGroundObs = dict[Grounding, list[GroundingValueType]]

class BaseGroundedModel(ABC):
    "Grounded model base class. This is primarily for special usecases when instance-specific information is needed, and is not required to use Vejde."

    @cached_property
    @abstractmethod
    def groundings(self) -> tuple[Grounding, ...]:
        """
        A list of all possible grounded facts based on the problem instance.
        on the form: (relation, object1, object2,..., objectN)
        """
        ...

    @cached_property
    @abstractmethod
    def action_groundings(self) -> tuple[Grounding, ...]:
        """groundings of action fluents/variables.
        on the form: (relation, object1, object2,..., objectN)
        """
        ...

    @cached_property
    @abstractmethod
    def constant_groundings(self) -> tuple[Grounding, ...]:
        """Groundings assumed to be constant in the model."""
        ...

    @cache
    @abstractmethod
    def constant_value(self, constant_grounding: Grounding) -> GroundingValueType:
        """The constant value of a constant grounding."""
        ...
