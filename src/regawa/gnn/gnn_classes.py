from collections.abc import Callable
from typing import Generic, NamedTuple, TypeVar

import numpy as np
import torch.nn.init as init
from numpy.typing import NDArray
from torch import Generator as Rngs
from torch import Tensor, concatenate
from torch.nn import Embedding, LayerNorm, Linear, Module, Sequential

V = TypeVar("V")


class SparseArray(NamedTuple, Generic[V]):
    values: NDArray[V]
    indices: NDArray[np.int64]

    @property
    def shape(self):
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
    def shape(self):
        return self.values.shape

    def concat(self, other: "SparseTensor") -> "SparseTensor":
        return SparseTensor(
            concatenate((self.values, other.values)),
            concatenate((self.indices, other.indices)),
        )


def sparsify(
    operation: Callable[[Tensor], Tensor],
) -> Callable[[SparseTensor], SparseTensor]:
    def wrapper(x: SparseTensor) -> SparseTensor:
        return SparseTensor(operation(x.values), x.indices)

    return wrapper


class EmbeddingLayer(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        rngs: Rngs,
        use_layer_norm: bool = True,
        use_padding: bool = True,
    ):
        super().__init__()  # type: ignore
        embedding = (
            Embedding(num_embeddings, embedding_dim, padding_idx=0)
            if use_padding
            else Embedding(num_embeddings, embedding_dim)
        )
        init.orthogonal_(embedding.weight)  # type: ignore

        if use_padding:
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
        rngs: Rngs,
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
