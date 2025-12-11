"""
PyTorch-specific data structures and conversion functions for factor graphs.
"""

from __future__ import annotations
from collections.abc import Callable
from typing import NamedTuple, TypeVar
import torch
from torch import (
    BoolTensor,
    FloatTensor,
    IntTensor,
    LongTensor,
    Size,
    Tensor,
    as_tensor,
    concatenate,
)


from .batch import BatchData, HeteroBatchData
from .sparse import SparseArray


def heterostatedata_to_tensors(
    data: HeteroBatchData, device: str | torch.device = "cpu"
) -> HeteroBatchData:
    return HeteroBatchData(
        statedata_to_tensors(data.boolean, device),
        statedata_to_tensors(data.numeric, device),
    )


V = TypeVar("V", torch.float32, torch.bool, torch.int64, torch.int8)  # type: ignore


class TorchBatchedVariables[T: Tensor](NamedTuple):
    var_value: SparseTensor[T]
    var_type: SparseTensor[T]
    n_variable: LongTensor
    length: LongTensor
    times: LongTensor


class TorchBatchedFactors(NamedTuple):
    factor: SparseTensor[LongTensor]
    n_factor: LongTensor


class TorchActionMask(NamedTuple):
    # mask that indicates which actions are valid for each factor, given the predicate type. Length matches factor.
    action_type_mask: BoolTensor
    # mask that indicates which actions are valid for each factor, given the predicate arity. Objects are not valid for predicates with no arguments. Length matches factor.
    action_arity_mask: BoolTensor


class TorchBatchData[T: Tensor](NamedTuple):
    """This represents a batch of multiple factor graphs."""

    factor: TorchBatchedFactors
    variables: TorchBatchedVariables[T]
    edges: TorchEdges
    global_variables: TorchBatchedVariables[T]
    action_masks: TorchActionMask
    n_graphs: int


class TorchHeteroBatchData(NamedTuple):
    """This represents a batch of multiple heterogeneous factor graphs."""

    boolean: TorchBatchData[IntTensor]
    numeric: TorchBatchData[FloatTensor]

    @property
    def n_graphs(self) -> int:
        return self.boolean.n_graphs


class TorchEdges(NamedTuple):
    # mappings from grounding to object. Length matches var_value
    v_to_f: LongTensor
    # mappings from object to grounding. Length matches factor
    f_to_v: LongTensor
    # edge attributes, e.g. position in predicate. Length matches v_to_f and f_to_v
    edge_attr: LongTensor


class SparseTensor[V: Tensor](NamedTuple):
    """
    This is a simple sparse COOrdinate array representation.
    index is the position of the values in the original dense array, e.g. the graph the node belongs to

    assuming [1, 2, 3], [4, 5] and [6, 7, 8, 9], the sparse representation will be
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    indices = [0, 0, 0, 1, 1, 2, 2, 2, 2]
    """

    values: Tensor
    indices: Tensor

    @property
    def shape(self) -> Size:
        return self.values.shape

    def map(self, func: Callable[[Tensor], Tensor]) -> SparseTensor[V]:
        return SparseTensor(
            func(self.values),
            self.indices,
        )

    def min(self, other: SparseTensor[V]) -> SparseTensor[V]:
        return SparseTensor(
            torch.min(self.values, other.values),
            self.indices,
        )

    def concat(self, other: SparseTensor[V]) -> SparseTensor[V]:
        return SparseTensor(
            concatenate((self.values, other.values)),
            concatenate((self.indices, other.indices)),
        )

    def replace_val(self, values: Tensor) -> SparseTensor[V]:
        if values.shape != self.values.shape:
            raise ValueError("New values must have the same shape as existing values")
        return self._replace(values=values)

    @classmethod
    def from_sparse_array(cls, sparse_array: SparseArray[V]) -> SparseTensor[V]:
        return cls(
            Tensor(sparse_array.values),
            Tensor(sparse_array.indices),
        )


def tuple_to_tensors[T: NamedTuple](
    data: NamedTuple, output_class: type[T], device: str | torch.device
) -> T:
    params = tuple(
        SparseTensor(
            as_tensor(attr.values, device=device),  # type: ignore
            as_tensor(attr.indices, device=device),
        )
        if isinstance(attr, SparseArray)
        else as_tensor(
            attr, device=device if key not in ("length", "global_length") else "cpu"
        )
        for key, attr in data._asdict().items()
    )
    return output_class(*params)


def statedata_to_tensors(
    data: BatchData[V], device: str | torch.device = "cpu"
) -> TorchBatchData[V]:
    return TorchBatchData(
        tuple_to_tensors(data.factor, TorchBatchedFactors, device),
        tuple_to_tensors(data.variables, TorchBatchedVariables, device),
        tuple_to_tensors(data.edges, TorchEdges, device),
        tuple_to_tensors(data.global_variables, TorchBatchedVariables, device),
        tuple_to_tensors(data.action_masks, TorchActionMask, device),
        as_tensor(data.n_graphs, device="cpu"),
    )


def sparsify(
    operation: Callable[[Tensor], Tensor],
) -> Callable[[SparseTensor[V]], SparseTensor[V]]:
    def wrapper(x: SparseTensor) -> SparseTensor:
        return SparseTensor(operation(x.values), x.indices)

    return wrapper


class TorchFactorGraph(NamedTuple):
    """
    This represents a single factor graph, with all features mapped to a vector space.
    """

    variables: SparseTensor[FloatTensor]
    factors: SparseTensor[FloatTensor]
    globals: SparseTensor[FloatTensor]
    v_to_f: Tensor
    f_to_v: Tensor
    edge_attr: Tensor
    n_variable: Tensor
    n_factor: Tensor


# @torch.jit.script  # type: ignore
def concat_sparse[V: Tensor](a: SparseTensor[V], b: SparseTensor[V]) -> SparseTensor[V]:
    return SparseTensor(
        concatenate((a.values, b.values)),
        concatenate((a.indices, b.indices)),
    )
