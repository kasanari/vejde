from typing import Generic, NamedTuple, TypeVar

import numpy as np


from numpy.typing import NDArray
from regawa.model import GroundValue

V = TypeVar("V", np.float32, np.bool_)


class Object(NamedTuple):
    name: str
    type: str


class Edge(NamedTuple):
    grounding: GroundValue
    object: str
    pos: int


class RenderGraph(NamedTuple):
    variable_labels: list[str]
    factor_labels: list[str]
    senders: NDArray[np.int64]
    receivers: NDArray[np.int64]
    edge_attributes: list[int]
    global_variables: list[str]


class Variables(NamedTuple, Generic[V]):
    types: NDArray[np.int64] | list[str]
    values: NDArray[V] | list[V]
    lengths: NDArray[np.int64] | list[int]


class IdxFactorGraph(NamedTuple, Generic[V]):
    variables: Variables[V]
    factors: NDArray[np.int64]
    senders: NDArray[np.int64]
    receivers: NDArray[np.int64]
    edge_attributes: NDArray[np.int64]
    global_vars: Variables[V]
    action_type_mask: NDArray[np.bool_]
    action_arity_mask: NDArray[np.bool_]


class FactorGraph(NamedTuple, Generic[V]):
    variables: list[str]
    variable_values: list[V]
    factors: list[str]
    factor_types: list[str]
    senders: NDArray[np.int64]
    receivers: NDArray[np.int64]
    edge_attributes: list[int]
    global_variables: list[str]
    global_variable_values: list[V]
    action_type_mask: list[tuple[bool, ...]]
    action_arity_mask: list[tuple[bool, ...]]
    groundings: list[GroundValue]
    global_groundings: list[GroundValue]


class StackedFactorGraph(NamedTuple, Generic[V]):
    variables: list[str]
    variable_values: list[list[V]]
    factors: list[str]
    factor_types: list[str]
    senders: NDArray[np.int64]
    receivers: NDArray[np.int64]
    edge_attributes: list[int]
    global_variables: list[str]
    global_variable_values: list[list[V]]
    action_mask: list[tuple[bool, ...]]
    groundings: list[GroundValue]
    global_groundings: list[GroundValue]


class HeteroGraph(NamedTuple):
    numeric: FactorGraph[np.float32] | StackedFactorGraph[np.float32]
    boolean: FactorGraph[np.bool_] | StackedFactorGraph[np.bool_]
