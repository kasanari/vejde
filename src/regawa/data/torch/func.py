from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import Tensor, as_tensor

from regawa.data.batch import Batch, HeteroBatch
from regawa.data.sparse import SparseArray

from .torch import (
    SparseTensor,
    TorchActionMask,
    TorchBatchData,
    TorchBatchedFactors,
    TorchBatchedVariables,
    TorchEdges,
    TorchHeteroBatchData,
)

V = torch.dtype | str | int | float | bool


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
    return output_class(*params)  # type: ignore


def statedata_to_tensors(
    data: Batch[V], device: str | torch.device = "cpu"
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
    def wrapper(x: SparseTensor[V]) -> SparseTensor[V]:
        return SparseTensor(operation(x.values), x.indices)

    return wrapper


def heterostatedata_to_tensors(
    data: HeteroBatch, device: str | torch.device = "cpu"
) -> TorchHeteroBatchData:
    return TorchHeteroBatchData(
        statedata_to_tensors(data.boolean, device),
        statedata_to_tensors(data.numeric, device),
    )
