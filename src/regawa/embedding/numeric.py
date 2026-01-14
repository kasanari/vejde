from torch import Tensor, nn

from regawa.embedding.node_embedders import EmbeddingLayer


class NumericEmbedder(nn.Module):
    """Embedder that multiplies the numeric value by the predicate embeddings."""

    def __init__(
        self,
        _embedding_dim: int,
        _activation: nn.Module,
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
        return preds * var_val.unsqueeze(-1)
