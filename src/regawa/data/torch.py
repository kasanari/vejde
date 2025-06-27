from __future__ import annotations
from collections.abc import Callable
from typing import NamedTuple, TypeVar
import torch
from torch import Tensor, as_tensor, concatenate
from .data import SparseArray, BatchData, HeteroBatchData

V = TypeVar("V")


def heterostatedata_to_tensors(
    data: HeteroBatchData, device: str | torch.device = "cpu"
) -> HeteroBatchData:
    return HeteroBatchData(
        statedata_to_tensors(data.boolean, device),
        statedata_to_tensors(data.numeric, device),
    )


class SparseTensor(NamedTuple):
    values: Tensor
    indices: Tensor

    @property
    def shape(self):
        return self.values.shape

    def concat(self, other: SparseTensor) -> SparseTensor:
        return SparseTensor(
            concatenate((self.values, other.values)),
            concatenate((self.indices, other.indices)),
        )

    @classmethod
    def from_sparse_array(cls, sparse_array: SparseArray) -> SparseTensor:
        return cls(
            Tensor(sparse_array.values),
            Tensor(sparse_array.indices),
        )


def statedata_to_tensors(
    data: BatchData, device: str | torch.device = "cpu"
) -> BatchData:
    params = tuple(
        SparseTensor(
            as_tensor(attr.values, device=device),
            as_tensor(attr.indices, device=device),
        )
        if isinstance(attr, SparseArray)
        else as_tensor(attr, device=device)
        for attr in data
    )

    return BatchData(*params)


def sparsify(
    operation: Callable[[Tensor], Tensor],
) -> Callable[[SparseTensor], SparseTensor]:
    def wrapper(x: SparseTensor) -> SparseTensor:
        return SparseTensor(operation(x.values), x.indices)

    return wrapper


class FactorGraph(NamedTuple):
    variables: SparseTensor
    factors: SparseTensor
    globals: SparseTensor
    v_to_f: Tensor
    f_to_v: Tensor
    edge_attr: Tensor
    n_variable: Tensor
    n_factor: Tensor
