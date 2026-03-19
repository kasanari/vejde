import logging
from functools import partial

from gnn_policy.functional import segment_sum, segmented_softmax
from torch import Generator, Tensor, nn

from regawa.data import SparseTensor
from regawa.nn import linear_reset_parameters

logger = logging.getLogger(__name__)

render_logger = logging.getLogger("message_pass_render")


class AttentionalAggregation(nn.Module):
    def __init__(self, emb_size: int, rngs: Generator):
        super().__init__()  # type: ignore

        init: partial[nn.Linear] = partial(linear_reset_parameters, rng=rngs)  # type: ignore
        self.gate = init(nn.Linear(emb_size, 1))
        self.attn = init(nn.Linear(emb_size, emb_size))

        logger.info("Attentional Aggregation\n")
        logger.info("Gate\n%s", self.gate)
        logger.info("Attention\n%s", self.attn)

    def forward(self, nodes: SparseTensor, num_graphs: int) -> Tensor:
        x = self.gate(nodes.values)
        x = segmented_softmax(x, nodes.indices, num_graphs)
        x = x * self.attn(nodes.values)
        return segment_sum(x, nodes.indices, num_graphs)
