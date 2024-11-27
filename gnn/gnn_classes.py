import torch.nn.init as init
from torch import Tensor
from torch.nn import Embedding, Linear, Module


class EmbeddingLayer(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, activation: Module):
        super().__init__()  # type: ignore
        self.embedding = Embedding(num_embeddings + 1, embedding_dim, padding_idx=0)
        init.orthogonal_(self.embedding.weight)  # type: ignore
        # init.ones_(self.embedding.weight)

        # self.bias = Parameter(torch.zeros(num_embeddings, embedding_dim))
        # self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        # bias = self.bias[x]
        return self.embedding(x)


class MLPLayer(Module):
    def __init__(self, in_features: int, out_features: int, activation: Module):
        super().__init__()  # type: ignore
        linear = Linear(in_features, out_features)
        init.orthogonal_(linear.weight)  # type: ignore
        init.constant_(linear.bias, 0.0)
        self.linear = linear
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.linear(x))
