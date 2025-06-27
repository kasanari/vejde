import logging

import torch.nn as nn
import torch.nn.init as init
from torch.nn import Embedding, LayerNorm, Module, Sequential
from torch import Generator as Rngs
from torch import Tensor, cumsum, roll, zeros
from torch.nn.utils.rnn import pack_padded_sequence

logger = logging.getLogger(__name__)


def plot_embeddings(num_embeddings: int, embedding: Embedding):
    import matplotlib.pyplot as plt

    for i in range(num_embeddings):
        plt.scatter(embedding.weight[i][0].item(), embedding.weight[i][1].item())
        # draw arrow from origin to point (x,y)
        plt.arrow(
            0,
            0,
            embedding.weight[i][0].item(),
            embedding.weight[i][1].item(),
            head_width=0.01,
            head_length=0.01,
            fc="r",
            ec="r",
        )
    plt.savefig("embedding.png")
    plt.close()


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


def compress_time(recurrent: nn.GRU, h: Tensor, length: Tensor) -> Tensor:
    padded = zeros(
        length.size(0),
        length.max().item(),
        h.size(-1),
    )

    offsets = roll(cumsum(length, axis=0), 1, 0)
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
        rngs: Rngs,
    ):
        super().__init__()  # type: ignore

        self.predicate_embedding = predicate_embedding
        logger.debug(
            "predicate_embedding:\n%s", predicate_embedding.transform[0].weight
        )

        self.boolean_embedding = EmbeddingLayer(
            2, embedding_dim, rngs, use_padding=False
        )

        logger.debug(
            "boolean_embedding:\n%s", self.boolean_embedding.transform[0].weight
        )

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


class NegativeBiasBooleanEmbedder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        predicate_embedding: EmbeddingLayer,
        rngs: Rngs,
    ):
        super().__init__()  # type: ignore

        self.predicate_embedding = predicate_embedding
        logger.debug(
            "predicate_embedding:\n%s", predicate_embedding.transform[0].weight
        )
        num_predicates = predicate_embedding.transform[0].weight.size(0)
        self.bias = nn.Parameter(zeros(num_predicates, embedding_dim))

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


class PositiveNegativeBooleanEmbedder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        predicate_embedding: EmbeddingLayer,
        rngs: Rngs,
    ):
        super().__init__()  # type: ignore

        self.positive_predicate_embedding = predicate_embedding
        logger.debug(
            "predicate_embedding:\n%s", predicate_embedding.transform[0].weight
        )
        num_predicates = predicate_embedding.transform[0].weight.size(0)
        self.neg_predicate_embedding = EmbeddingLayer(
            num_predicates, embedding_dim, rngs
        )

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


class NumericEmbedder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        activation: nn.Module,
        predicate_embedding: EmbeddingLayer,
    ):
        super().__init__()  # type: ignore

        self.predicate_embedding = predicate_embedding

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

            return variables.squeeze(0)

        return _forward
