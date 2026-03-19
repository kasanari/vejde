from __future__ import annotations

from collections.abc import Iterable
from typing import Generic, NamedTuple, TypeVar

import numpy as np
from numpy.typing import NDArray

from regawa.data.graph import VariableDomain
from regawa.data.graph.graph import ActionMask, Edges
from regawa.data.sparse import SparseArray

from ..obs import HeteroIndexedFactorGraph, IndexedFactorGraph
from .batch_func import batch, batch_from_buffer


class BatchedVariables(NamedTuple, Generic[VariableDomain]):
    var_value: SparseArray[VariableDomain]
    var_type: SparseArray[np.int64]
    n_variable: NDArray[np.int64]
    length: NDArray[np.int64]
    times: NDArray[np.int64]


class BatchedFactors(NamedTuple):
    factor: SparseArray[np.int64]
    n_factor: NDArray[np.int64]


class Batch(NamedTuple, Generic[VariableDomain]):
    """This represents a batch of multiple factor graphs."""

    factor: BatchedFactors
    variables: BatchedVariables[VariableDomain]
    edges: Edges
    n_graphs: np.int64
    global_variables: BatchedVariables[VariableDomain]
    action_masks: ActionMask


class HeteroBatch(NamedTuple):
    """This represents a batch of multiple heterogeneous factor graphs."""

    boolean: Batch[np.int8]
    numeric: Batch[np.float32]

    @property
    def n_graphs(self) -> np.int64:
        return self.boolean.n_graphs

    @property
    def n_factor(self) -> NDArray[np.int64]:
        # The assumption at the moment is that both boolean and numeric use the same factors, even if one might have no variables.
        return self.boolean.factor.n_factor

    @classmethod
    def from_obs(cls, obs: Iterable[HeteroIndexedFactorGraph]) -> HeteroBatch:
        return heterobatch(obs)

    @classmethod
    def from_buffer(
        cls, obs: dict[str, list[tuple[IndexedFactorGraph[VariableDomain], ...]]]
    ) -> HeteroBatch:
        return heterostatedata_from_buffer(obs)

    @classmethod
    def from_single_obs(cls, obs: HeteroIndexedFactorGraph) -> HeteroBatch:
        return single_obs_to_heterostatedata(obs)


ArrayDomain = TypeVar("ArrayDomain", np.int8, np.float32, np.bool_, np.int64)


def heterobatch(
    obs: Iterable[HeteroIndexedFactorGraph],
) -> HeteroBatch:
    return HeteroBatch(
        boolean=batch([o.bool for o in obs]),
        numeric=batch([o.float for o in obs]),
    )


def heterostatedata_from_buffer(
    obs: dict[str, list[tuple[IndexedFactorGraph[VariableDomain], ...]]],
) -> HeteroBatch:
    return HeteroBatch(
        boolean=batch_from_buffer(obs["bool"]),  # type: ignore
        numeric=batch_from_buffer(obs["float"]),  # type: ignore
    )


def single_obs_to_heterostatedata(obs: HeteroIndexedFactorGraph) -> HeteroBatch:
    return heterobatch([obs])
