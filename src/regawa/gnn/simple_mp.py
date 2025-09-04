import logging

from torch import Generator as Rngs
from torch import Tensor, concatenate, nn
from torch_scatter import scatter  # type: ignore

from regawa.gnn.gnn_classes import MLPLayer

logger = logging.getLogger(__name__)


class MessagePass(nn.Module):
    """
    A simple message passing layer with customizable aggregation. Uses scatter for aggregation.

    The resulting vector will be of shape (num_segments, out_channels). The recipients vector
    indicates which position in the output each message will be aggregated into.

    Args:
    x_i: Target node features (Tensor of shape [num_target_nodes, in_channels])
    x_j: Source node features (Tensor of shape [num_source_nodes, in_channels])
    recipients: Indices of target nodes for each message (Tensor of shape [num_messages]). Values should be in the range [0, num_target_nodes - 1]
    edge_attr: Edge features (Tensor of shape [num_messages, edge_attr_dim])
    num_segments: Number of target nodes (int)

    Returns:
        aggr_m: Aggregated messages for each target node (Tensor of shape [num_target_nodes, out_channels])
        m: Individual messages (Tensor of shape [num_messages, out_channels])
    """

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
        x_i: Tensor,  # target nodes
        x_j: Tensor,  # source nodes
        recipients: Tensor,  # indices of target nodes
        edge_attr: Tensor,  # edge features
        num_segments: int,  # number of target nodes
    ):
        x = concatenate(
            (x_i, x_j, edge_attr), dim=-1
        )  # message is a function of target, source, and edge. This could be changed.
        m = self.message_func(x)
        aggr_m = scatter(m, recipients, dim=0, reduce=self.aggr, dim_size=num_segments)
        logger.debug("X_j\n%s", x_j)
        logger.debug("X_i\n%s", x_i)
        logger.debug("Messages\n%s", m)
        logger.debug("Aggr out\n%s", aggr_m)
        return aggr_m, m
