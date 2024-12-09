import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence
import torch as th
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
        self.boolean_embedding = EmbeddingLayer(2, embedding_dim, activation)

    def forward(
        self,
        var_val: Tensor,
        var_type: Tensor,
        object_class: Tensor,
    ) -> tuple[Tensor, Tensor]:
        booleans = self.boolean_embedding(var_val.int())
        preds = self.predicate_embedding(var_type.int())

        # indices = (var_val * var_type).int()
        # h_p = self.predicate_embedding(indices)
        h_p = booleans * preds
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

        self.boolean_embedding = EmbeddingLayer(2, embedding_dim, activation)

        self.activation = activation

        self.recurrent = nn.GRU(embedding_dim, embedding_dim, batch_first=True)

        for name, param in self.recurrent.named_parameters():
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
        bools = self.boolean_embedding(var_val.int())
        preds = self.predicate_embedding(var_type.int())

        h = bools * preds

        padded = th.zeros(
            length.size(0),
            length.max().item(),
            h.size(-1),
        )
        offset = 0
        for i, l in enumerate(length):
            padded[i, : l.item()] = h[offset : offset + l.item()]
            offset += l.item()

        # indices = (var_val * var_type.unsqueeze(1)).int()
        # e = self.predicate_embedding(padded)
        h_c = pack_padded_sequence(
            padded, length, batch_first=True, enforce_sorted=False
        )
        _, variables = self.recurrent(h_c)

        return variables.squeeze(), factors
