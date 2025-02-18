from typing import Any, Generic, NamedTuple, TypeVar

import numpy as np

from regawa.model import GroundValue

V = TypeVar("V")


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
    senders: np.ndarray[np.int64, Any]
    receivers: np.ndarray[np.int64, Any]
    edge_attributes: list[int]
    global_variables: list[str]


class Variables(NamedTuple, Generic[V]):
    types: np.ndarray[np.int64, Any] | list[str]
    values: np.ndarray[V, Any] | list[V]
    lengths: np.ndarray[np.int64, Any] | list[int]


class IdxFactorGraph(NamedTuple, Generic[V]):
    variables: Variables[V]
    factors: np.ndarray[np.int64, Any]
    senders: np.ndarray[np.int64, Any]
    receivers: np.ndarray[np.int64, Any]
    edge_attributes: np.ndarray[np.int64, Any]
    global_vars: Variables[V]
    action_mask: np.ndarray[np.bool_, Any]


class FactorGraph(NamedTuple, Generic[V]):
    variables: list[str]
    variable_values: list[V]
    factors: list[str]
    factor_types: list[str]
    senders: np.ndarray[np.int64, Any]
    receivers: np.ndarray[np.int64, Any]
    edge_attributes: list[int]
    global_variables: list[str]
    global_variable_values: list[V]
    action_mask: list[tuple[bool, ...]]
    groundings: list[GroundValue]
    global_groundings: list[GroundValue]


class StackedFactorGraph(NamedTuple, Generic[V]):
    variables: list[str]
    variable_values: list[list[V]]
    factors: list[str]
    factor_types: list[str]
    senders: np.ndarray[np.int64, Any]
    receivers: np.ndarray[np.int64, Any]
    edge_attributes: list[int]
    global_variables: list[str]
    global_variable_values: list[list[V]]
    action_mask: list[tuple[bool, ...]]
    groundings: list[GroundValue]
    global_groundings: list[GroundValue]


class HeteroGraph(NamedTuple):
    numeric: FactorGraph[float] | StackedFactorGraph[float]
    boolean: FactorGraph[bool] | StackedFactorGraph[bool]
