from torch.nn import Module, Embedding, Linear, Parameter
from torch import Tensor
import torch.nn as nn
import torch.nn.init as init
import torch as th
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    init.orthogonal_(layer.weight)
    init.constant_(layer.bias, bias_const)
    return layer


class EmbeddingLayer(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, activation: Module):
        super(EmbeddingLayer, self).__init__()
        self.embedding = Embedding(num_embeddings + 1, embedding_dim, padding_idx=0)
        # init.orthogonal_(self.embedding.weight)
        # init.ones_(self.embedding.weight)

        # self.bias = Parameter(torch.zeros(num_embeddings, embedding_dim))
        # self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        # bias = self.bias[x]
        return self.embedding(x)


class MLPLayer(Module):
    def __init__(self, in_features: int, out_features: int, activation: Module):
        super(MLPLayer, self).__init__()
        self.linear = layer_init(Linear(in_features, out_features))
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.linear(x))
