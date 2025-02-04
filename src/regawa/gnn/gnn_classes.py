from collections.abc import Callable
from typing import NamedTuple, TypeVar
import torch.nn.init as init
from torch import Tensor
import torch
from torch.nn import Embedding, Linear, Module, LayerNorm, Sequential
from numpy.typing import NDArray
import numpy as np

V = TypeVar("V")


class SparseArray[V](NamedTuple):
    values: NDArray[V]
    indices: NDArray[np.int64]

    @property
    def shape(self) -> torch.Size:
        return self.values.shape

    def concat(self, other: "SparseTensor") -> "SparseTensor":
        return SparseTensor(
            np.concatenate((self.values, other.values)),
            np.concatenate((self.indices, other.indices)),
        )


class SparseTensor(NamedTuple):
    values: Tensor
    indices: Tensor

    @property
    def shape(self) -> torch.Size:
        return self.values.shape

    def concat(self, other: "SparseTensor") -> "SparseTensor":
        return SparseTensor(
            torch.cat((self.values, other.values)),
            torch.cat((self.indices, other.indices)),
        )


def sparsify(
    operation: Callable[[Tensor], Tensor],
) -> Callable[[SparseTensor], SparseTensor]:
    def wrapper(x: SparseTensor) -> SparseTensor:
        return SparseTensor(operation(x.values), x.indices)

    return wrapper


class EmbeddingLayer(Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, use_layer_norm: bool = True
    ):
        super().__init__()  # type: ignore
        embedding = Embedding(num_embeddings, embedding_dim, padding_idx=0)
        init.orthogonal_(embedding.weight)  # type: ignore
        embedding._fill_padding_idx_with_zero()
        layer_norm = LayerNorm(embedding_dim, elementwise_affine=True)
        # init.ones_(self.embedding.weight)

        # self.bias = Parameter(torch.zeros(num_embeddings, embedding_dim))
        # self.activation = activation

        params = (embedding, layer_norm) if use_layer_norm else (embedding,)

        self.transform = Sequential(
            *params,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transform(x)


class MLPLayer(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Module,
        use_layer_norm: bool = False,
    ):
        super().__init__()  # type: ignore
        linear = Linear(in_features, out_features)
        init.orthogonal_(linear.weight)  # type: ignore
        init.constant_(linear.bias, 0.0)

        layer_norm = LayerNorm(out_features, elementwise_affine=False)

        layers = (
            (linear, layer_norm, activation) if use_layer_norm else (linear, activation)
        )

        self.transform = Sequential(
            *layers,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transform(x)

    def __str__(self) -> str:
        return f"Weight:\n{self.transform[0].weight}\nBias:\n{self.transform[0].bias}"
