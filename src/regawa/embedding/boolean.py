from regawa.embedding.node_embedders import EmbeddingLayer, logger


import torch.nn as nn
from torch import Generator as Rngs, Tensor, zeros


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