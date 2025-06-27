import logging

from torch import Generator as Rngs
from torch import Tensor, concatenate, nn
from torch_scatter import scatter  # type: ignore

from .mlp import MLPLayer

logger = logging.getLogger(__name__)


class MessagePass(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggregation: str,
        activation: nn.Module,
        rngs: Rngs,
    ):
        super().__init__()  # type: ignore
        self.message_func = MLPLayer(in_channels * 3, out_channels, activation, rngs)
        self.aggr = aggregation

    def forward(
        self,
        x_i: Tensor,
        x_j: Tensor,
        recipients: Tensor,
        edge_attr: Tensor,
        num_segments: int,
    ):
        x = concatenate((x_i, x_j, edge_attr), dim=-1)
        m = self.message_func(x)
        aggr_m = scatter(m, recipients, dim=0, reduce=self.aggr, dim_size=num_segments)
        logger.debug("X_j\n%s", x_j)
        logger.debug("X_i\n%s", x_i)
        logger.debug("Messages\n%s", m)
        logger.debug("Aggr out\n%s", aggr_m)
        return aggr_m, m
