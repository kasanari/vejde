import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence

from .gnn_classes import EmbeddingLayer


class Embedder(nn.Module):
    def __init__(
        self,
        num_object_classes: int,
        num_predicate_classes: int,
        embedding_dim: int,
        activation: nn.Module,
    ):
        super().__init__()  # type: ignore

        self.object_embedding = EmbeddingLayer(
            num_object_classes, embedding_dim, activation
        )
        self.predicate_embedding = EmbeddingLayer(
            num_predicate_classes, embedding_dim, activation
        )

    def forward(
        self,
        var_val: Tensor,
        var_type: Tensor,
        object_class: Tensor,
    ) -> tuple[Tensor, Tensor]:
        indices = (var_val * var_type).int()
        h_p = self.predicate_embedding(indices)
        h_o = self.object_embedding(object_class)
        return h_p, h_o


class RecurrentEmbedder(nn.Module):
    def __init__(
        self,
        num_object_classes: int,
        num_predicate_classes: int,
        embedding_dim: int,
        activation: nn.Module,
    ):
        super().__init__()  # type: ignore

        self.object_embedding = EmbeddingLayer(
            num_object_classes, embedding_dim, activation
        )
        self.predicate_embedding = EmbeddingLayer(
            num_predicate_classes, embedding_dim, activation
        )

        self.lstm = nn.LSTM(embedding_dim, embedding_dim, 1, batch_first=True)

        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                init.orthogonal_(param)  # type: ignore
            elif "bias" in name:
                init.zeros_(param)

    def forward(
        self,
        var_val: Tensor,
        var_type: Tensor,
        factor_type: Tensor,
        length: Tensor,
    ):
        factors = self.object_embedding(factor_type)
        indices = (var_val * var_type.unsqueeze(1)).int()
        h_c = self.predicate_embedding(indices)
        h_c = pack_padded_sequence(h_c, length, batch_first=True, enforce_sorted=False)
        _, (variables, _) = self.lstm(h_c)

        return variables.squeeze(), factors
