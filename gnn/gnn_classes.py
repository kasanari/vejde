from torch.nn import Module, Embedding, Linear
from torch import Tensor
import torch.nn as nn
import torch.nn.init as init
import numpy as np


class EmbeddingLayer(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, activation: Module):
        super(EmbeddingLayer, self).__init__()
        self.embedding = Embedding(num_embeddings + 1, embedding_dim, padding_idx=0)
        init.orthogonal_(self.embedding.weight)
        # init.ones_(self.embedding.weight)

        # self.bias = Parameter(torch.zeros(num_embeddings, embedding_dim))
        # self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        # bias = self.bias[x]
        return self.embedding(x)


class MLPLayer(Module):
    def __init__(self, in_features: int, out_features: int, activation: Module):
        super(MLPLayer, self).__init__()
        linear = Linear(in_features, out_features)
        init.orthogonal_(linear.weight)
        init.constant_(linear.bias, 0.0)
        self.linear = linear
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.linear(x))
