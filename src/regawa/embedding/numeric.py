from regawa.embedding.node_embedders import EmbeddingLayer


import torch.nn as nn
from torch import Tensor


class NumericEmbedder(nn.Module):
    """Embedder that multiplies the numeric value by the predicate embeddings."""

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
