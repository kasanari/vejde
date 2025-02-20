import logging

import torch as th
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence

from .gnn_classes import EmbeddingLayer

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


class BooleanEmbedder(th.jit.ScriptModule):
    def __init__(
        self,
        embedding_dim: int,
        predicate_embedding: EmbeddingLayer,
    ):
        super().__init__()  # type: ignore

        self.predicate_embedding = predicate_embedding
        logger.debug(
            "predicate_embedding:\n%s", predicate_embedding.transform[0].weight
        )

        self.boolean_embedding = EmbeddingLayer(2, embedding_dim, use_padding=False)

        logger.debug(
            "boolean_embedding:\n%s", self.boolean_embedding.transform[0].weight
        )

    @th.jit.script_method
    def forward(
        self,
        var_val: Tensor,
        var_type: Tensor,
    ) -> Tensor:
        booleans = self.boolean_embedding(var_val.int())
        preds = self.predicate_embedding(var_type.int())
        # logger.debug("bools:\n%s", booleans)
        # logger.debug("preds:\n%s", preds)
        h = booleans * preds
        return h


class NegativeBiasBooleanEmbedder(th.jit.ScriptModule):
    def __init__(
        self,
        embedding_dim: int,
        predicate_embedding: EmbeddingLayer,
    ):
        super().__init__()  # type: ignore

        self.predicate_embedding = predicate_embedding
        logger.debug(
            "predicate_embedding:\n%s", predicate_embedding.transform[0].weight
        )
        num_predicates = predicate_embedding.transform[0].weight.size(0)
        self.bias = nn.Parameter(th.zeros(num_predicates, embedding_dim))

    @th.jit.script_method
    def forward(
        self,
        var_val: Tensor,
        var_type: Tensor,
    ) -> Tensor:
        preds = self.predicate_embedding(var_type.int())
        biases = self.bias[var_type.int()]
        # logger.debug("bools:\n%s", booleans)
        # logger.debug("preds:\n%s", preds)
        h = var_val.unsqueeze(1) * preds + biases
        return h


class PositiveNegativeBooleanEmbedder(th.jit.ScriptModule):
    def __init__(
        self,
        embedding_dim: int,
        predicate_embedding: EmbeddingLayer,
    ):
        super().__init__()  # type: ignore

        self.positive_predicate_embedding = predicate_embedding
        logger.debug(
            "predicate_embedding:\n%s", predicate_embedding.transform[0].weight
        )
        num_predicates = predicate_embedding.transform[0].weight.size(0)
        self.neg_predicate_embedding = EmbeddingLayer(num_predicates, embedding_dim)

    @th.jit.script_method
    def forward(
        self,
        var_val: Tensor,
        var_type: Tensor,
    ) -> Tensor:
        postive_preds = self.positive_predicate_embedding(var_type.int())
        negative_preds = self.neg_predicate_embedding(var_type.int())

        # logger.debug("bools:\n%s", booleans)
        # logger.debug("preds:\n%s", preds)
        h = (
            var_val.unsqueeze(1) * postive_preds
            + (1 - var_val).unsqueeze(1) * negative_preds
        )
        return h


class NumericEmbedder(th.jit.ScriptModule):
    def __init__(
        self,
        embedding_dim: int,
        activation: nn.Module,
        predicate_embedding: EmbeddingLayer,
    ):
        super().__init__()  # type: ignore

        self.predicate_embedding = predicate_embedding

    @th.jit.script_method
    def forward(
        self,
        var_val: Tensor,
        var_type: Tensor,
    ) -> Tensor:
        preds = self.predicate_embedding(var_type.int())
        h = preds * var_val.unsqueeze(-1)
        return h


class RecurrentEmbedder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        base_embedder: NegativeBiasBooleanEmbedder | NumericEmbedder | BooleanEmbedder,
    ):
        super().__init__()  # type: ignore

        self.embedder = base_embedder

        self.recurrent = nn.RNN(
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

            variables = (
                compress_time(self.recurrent, h, length) if h.shape[0] > 0 else h
            )

            logger.debug("variables:\n%s", variables)

            return variables.squeeze()

        return _forward
