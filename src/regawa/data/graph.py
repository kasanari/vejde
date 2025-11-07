from __future__ import annotations
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from regawa.data.actions import ActionMask
from regawa.model import Grounding


from typing import Generic, NamedTuple


from typing import TypeVar
from regawa.model.null import NullConst

VariableDomain = TypeVar(
    "VariableDomain",
    np.float32,
    np.bool_,
    np.int8,
)


VariableTypeDomain = np.int64
EdgeIndexDomain = np.int64


class Distances(NamedTuple):
    source_variables: NDArray[np.int64]
    target_factors: NDArray[np.int64]
    distances: NDArray[np.float32]

    # distances: Distances


class Variables(NamedTuple, Generic[VariableDomain]):
    # predicate of grounding, e.g. "p". Length matches var_value.
    types: NDArray[VariableTypeDomain]
    # number of repetitions per grounding. This is only not 1 when using stacking. Length matches var_value
    # value of groundings, e.g. "v". This can be bool or float
    value: NDArray[VariableDomain]
    length: NDArray[VariableTypeDomain]
    # number of groundings/variables. Will match len(length), even with stacking. Will match len(var_value) without stacking.
    n_variable: int  # number of groundings/variables. Will match len(length), even with stacking. Will match len(var_value) without stacking.
    times: NDArray[np.int64]  # time steps of variables


class Edge(NamedTuple):
    grounding: Grounding
    object: str
    pos: int


class Object(NamedTuple):
    name: str
    type: str

NullObject = Object(NullConst.id, NullConst.type)

class StackedStringVariables(NamedTuple, Generic[VariableDomain]):
    types: Sequence[str]
    values: Sequence[Sequence[VariableDomain]]
    length: Sequence[int]
    n_variable: int
    groundings: Sequence[Grounding]


class StringVariables(NamedTuple, Generic[VariableDomain]):
    types: Sequence[str]
    values: Sequence[VariableDomain]
    length: Sequence[int]
    n_variable: int  # number of groundings/variables. Will match len(length),
    groundings: Sequence[Grounding]


class StringFactors(NamedTuple):
    names: Sequence[str]  # object names
    types: Sequence[str]  # object of grounding, e.g. "o"


class StringFactorGraph(NamedTuple, Generic[VariableDomain]):
    """A FactorGraph with string attributes."""

    variables: StringVariables[VariableDomain]
    factors: StringFactors
    edges: Edges
    global_variables: StringVariables[VariableDomain]
    action_masks: ActionMask
    # distance metrics, in a sparse format
    # distances: Distances


class StackedStringFactorGraph(NamedTuple, Generic[VariableDomain]):
    variables: StackedStringVariables[VariableDomain]
    factors: StringFactors
    edges: Edges
    global_variables: StackedStringVariables[VariableDomain]
    action_masks: ActionMask


class HeteroGraph(NamedTuple):
    numeric: StringFactorGraph[np.float32] | StackedStringFactorGraph[np.float32]
    boolean: StringFactorGraph[np.bool_] | StackedStringFactorGraph[np.bool_]


class Edges(NamedTuple):
    # mappings from grounding to object. Length matches var_value
    v_to_f: NDArray[EdgeIndexDomain]
    # mappings from object to grounding. Length matches factor
    f_to_v: NDArray[EdgeIndexDomain]
    # edge attributes, e.g. position in predicate. Length matches v_to_f and f_to_v
    edge_attr: NDArray[EdgeIndexDomain]


GraphTypes = TypeVar(
    "GraphTypes",
    StringFactorGraph[np.bool_],
    StackedStringFactorGraph[np.bool_],
    StringFactorGraph[np.float32],
    StackedStringFactorGraph[np.float32],
)
