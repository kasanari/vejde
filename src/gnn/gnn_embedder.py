import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence
import torch as th
from .gnn_classes import EmbeddingLayer
import logging

logger = logging.getLogger(__name__)


def compress_time(recurrent: nn.GRU, h: Tensor, length: Tensor) -> Tensor:
    padded = th.zeros(
        length.size(0),
        length.max().item(),
        h.size(-1),
    )

    offsets = th.roll(th.cumsum(length, dim=0), 1, 0)
    offsets[0] = 0
    for i, node_l in enumerate(length):
        padded[i, : node_l.item()] = h[offsets[i] : offsets[i] + node_l.item()]

    h_c = pack_padded_sequence(padded, length, batch_first=True, enforce_sorted=False)
    _, variables = recurrent(h_c)
    return variables


class BooleanEmbedder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        predicate_embedding: EmbeddingLayer,
    ):
        super().__init__()  # type: ignore

        self.predicate_embedding = predicate_embedding

        self.boolean_embedding = EmbeddingLayer(2, embedding_dim)

    def forward(
        self,
        var_val: Tensor,
        var_type: Tensor,
    ) -> Tensor:
        booleans = self.boolean_embedding(var_val.int())
        preds = self.predicate_embedding(var_type.int())
        logger.debug("bools:\n%s", booleans)
        logger.debug("preds:\n%s", preds)
        h = booleans * preds
        return h


class NumericEmbedder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        activation: nn.Module,
        predicate_embedding: EmbeddingLayer,
    ):
        super().__init__()  # type: ignore

        self.predicate_embedding = predicate_embedding

        self.biases = nn.Parameter(
            th.zeros(predicate_embedding.embedding.weight.shape[0], embedding_dim)
        )
        self.activation = activation

    def forward(
        self,
        var_val: Tensor,
        var_type: Tensor,
    ) -> Tensor:
        preds = self.predicate_embedding(var_type.int())
        biases = self.biases[var_type.int()]
        h = nn.functional.relu(preds * var_val.unsqueeze(-1) + biases)
        return h


class RecurrentEmbedder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        base_embedder: BooleanEmbedder | NumericEmbedder,
    ):
        super().__init__()  # type: ignore

        self.embedder = base_embedder

        self.recurrent = nn.GRU(
            embedding_dim,
            embedding_dim,
            batch_first=True,
        )

        for name, param in self.recurrent.named_parameters():
            if "weight" in name:
                init.orthogonal_(param)  # type: ignore
            elif "bias" in name:
                init.zeros_(param)

    def forward(
        self,
        length: Tensor,
    ):
        def _forward(
            var_val: Tensor,
            var_type: Tensor,
        ):
            h = self.embedder(var_val, var_type)

            logger.debug("h:\n%s", h)

            variables = compress_time(self.recurrent, h, length)

            logger.debug("variables:\n%s", variables)

            return variables.squeeze()

        return _forward
