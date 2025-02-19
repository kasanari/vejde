import torch.nn as nn
from torch import Tensor
import logging
from gnn_policy.functional import segment_sum, segmented_softmax
from .gnn_classes import SparseTensor

logger = logging.getLogger(__name__)

render_logger = logging.getLogger("message_pass_render")


class AttentionalAggregation(nn.Module):
    def __init__(self, emb_size: int):
        super().__init__()  # type: ignore

        self.gate = nn.Linear(emb_size, 1)
        self.attn = nn.Linear(emb_size, emb_size)

        logger.info("Attentional Aggregation\n")
        logger.info("Gate\n%s", self.gate)
        logger.info("Attention\n%s", self.attn)

    def forward(self, nodes: SparseTensor, num_graphs: int) -> Tensor:
        x = self.gate(nodes.values)
        x = segmented_softmax(x, nodes.indices, num_graphs)
        x = x * self.attn(nodes.values)
        x = segment_sum(x, nodes.indices, num_graphs)
        return x
