class GlobalNode(nn.Module):
    def __init__(self, emb_size: int, activation: nn.Module):
        super().__init__()  # type: ignore

        self.attn = AttentionalAggregation(emb_size, activation)
        self.linear = MLPLayer(emb_size * 2, emb_size, activation)

        logger.info("Global Node\n")
        logger.info("Update Function\n%s", self.linear)

        # self.aggr = SumAggregation()

    def forward(self, nodes: SparseTensor, g_prev: Tensor) -> Tensor:
        x = self.attn(nodes)
        x = (x, g_prev)
        x = concatenate(x, axis=-1)
        x = g_prev + self.linear(x)
        return x
