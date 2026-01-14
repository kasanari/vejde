from torch import Generator as Rngs
from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, Sequential, init


class MLPLayer(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Module,
        rngs: Rngs,
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
        self.rngs = rngs

    def forward(self, x: Tensor) -> Tensor:
        return self.transform(x)

    def __str__(self) -> str:
        return f"Weight:\n{self.transform[0].weight}\nBias:\n{self.transform[0].bias}"
