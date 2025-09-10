from collections.abc import Sequence
from typing import Generic, NamedTuple, TypeVar

import numpy as np


from numpy.typing import NDArray
from regawa.model import Grounding

V = TypeVar("V", np.float32, np.bool_)


class Object(NamedTuple):
    name: str
    type: str


class Edge(NamedTuple):
    grounding: Grounding
    object: str
    pos: int


class RenderGraph(NamedTuple):
    variable_labels: Sequence[str]
    factor_labels: Sequence[str]
    senders: NDArray[np.int64]
    receivers: NDArray[np.int64]
    edge_attributes: Sequence[int]
    global_variables: Sequence[str]


class Variables(NamedTuple, Generic[V]):
    types: Sequence[np.int64] | Sequence[str]
    values: Sequence[V]
    lengths: NDArray[np.int64] | Sequence[int]


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
    variables: Sequence[str]
    variable_values: Sequence[V]
    factors: Sequence[str]
    factor_types: Sequence[str]
    senders: NDArray[np.int64]
    receivers: NDArray[np.int64]
    edge_attributes: Sequence[int]
    global_variables: Sequence[str]
    global_variable_values: Sequence[V]
    action_type_mask: Sequence[tuple[bool, ...]]
    action_arity_mask: Sequence[tuple[bool, ...]]
    groundings: Sequence[Grounding]
    global_groundings: Sequence[Grounding]


class StackedFactorGraph(NamedTuple, Generic[V]):
    variables: Sequence[str]
    variable_values: Sequence[Sequence[V]]
    factors: Sequence[str]
    factor_types: Sequence[str]
    senders: NDArray[np.int64]
    receivers: NDArray[np.int64]
    edge_attributes: Sequence[int]
    global_variables: Sequence[str]
    global_variable_values: Sequence[Sequence[V]]
    action_type_mask: Sequence[tuple[bool, ...]]
    action_arity_mask: Sequence[tuple[bool, ...]]
    groundings: Sequence[Grounding]
    global_groundings: Sequence[Grounding]


class HeteroGraph(NamedTuple):
    numeric: FactorGraph[np.float32] | StackedFactorGraph[np.float32]
    boolean: FactorGraph[np.bool_] | StackedFactorGraph[np.bool_]
