import torch.nn.init as init
from torch import Tensor
from torch.nn import Embedding, Linear, Module, LayerNorm, Sequential


class EmbeddingLayer(Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, use_layer_norm: bool = False
    ):
        super().__init__()  # type: ignore
        embedding = Embedding(num_embeddings, embedding_dim, padding_idx=0)
        init.orthogonal_(embedding.weight)  # type: ignore
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


class MLPLayer(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Module,
        use_layer_norm: bool = False,
    ):
        super().__init__()  # type: ignore
        linear = Linear(in_features, out_features)
        init.orthogonal_(linear.weight)  # type: ignore
        init.constant_(linear.bias, 0.0)

        layer_norm = LayerNorm(out_features, elementwise_affine=False)

        layers = (
            (linear, layer_norm, activation) if use_layer_norm else (linear, activation)
        )

        self.transform = Sequential(
            *layers,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transform(x)

    def __str__(self) -> str:
        return f"Weight:\n{self.transform[0].weight}\nBias:\n{self.transform[0].bias}"
