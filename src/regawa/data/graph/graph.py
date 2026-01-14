from __future__ import annotations

from collections.abc import Sequence
from typing import Generic, NamedTuple, TypeVar

import numpy as np
from numpy.typing import NDArray

from regawa.model import Grounding, NullConst

VariableDomain = TypeVar(
    "VariableDomain",
    np.float32,
    np.bool_,
    np.int8,
)


VariableTypeDomain = np.int64
EdgeIndexDomain = np.int64


class ActionMask(NamedTuple):
    # mask that indicates which actions are valid for each factor, given the predicate type. Length matches factor.
    action_type_mask: NDArray[np.bool_]
    # mask that indicates which actions are valid for each factor, given the predicate arity. Objects are not valid for predicates with no arguments. Length matches factor.
    action_arity_mask: NDArray[np.bool_]


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


class StringVariables(NamedTuple, Generic[VariableDomain]):
    types: Sequence[str]
    values: Sequence[VariableDomain]
    length: Sequence[int]
    n_variable: int  # number of groundings/variables. Will match len(length),
    groundings: Sequence[Grounding]


class StringFactors(NamedTuple):
    names: Sequence[str]  # object names
    types: Sequence[str]  # object of grounding, e.g. "o"


class Edges(NamedTuple):
    # mappings from grounding to object. Length matches var_value
    v_to_f: NDArray[EdgeIndexDomain]
    # mappings from object to grounding. Length matches factor
    f_to_v: NDArray[EdgeIndexDomain]
    # edge attributes, e.g. position in predicate. Length matches v_to_f and f_to_v
    edge_attr: NDArray[EdgeIndexDomain]

    # distance metrics, in a sparse format
    # distances: Distances
