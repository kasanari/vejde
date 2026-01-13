from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from functools import cache, cached_property

import numpy as np

type Grounding = tuple[str, ...]
type TemporalGrounding = tuple[int, Grounding]
type GroundingRange = bool | int | float | np.bool_
type ObservableGroundingRange = GroundingRange | None
GroundObs = Mapping[Grounding, GroundingRange]
type ObservableGroundObs = Mapping[Grounding, ObservableGroundingRange]

type TemporalGroundObs = Mapping[TemporalGrounding, GroundingRange]
StackedGroundObs = Mapping[Grounding, Sequence[GroundingRange]]


class BaseGroundedModel(ABC):
    "Grounded model base class. This is primarily for special usecases when instance-specific information is needed, and is not required to use Vejde."

    @cached_property
    @abstractmethod
    def objects(self) -> tuple[str, ...]:
        """A list of all objects in the instance."""
        ...

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
    def constant_value(self, constant_grounding: Grounding) -> GroundingRange:
        """The constant value of a constant grounding."""
        ...
