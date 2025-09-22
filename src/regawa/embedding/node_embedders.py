import logging

import torch.nn.init as init
from torch.nn import Embedding, LayerNorm, Module, Sequential
from torch import Generator as Rngs
from torch import Tensor

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
