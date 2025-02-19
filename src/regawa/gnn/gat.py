from torch_scatter import scatter  # type: ignore
from torch import nn, Tensor, empty

from gnn_policy.functional import segmented_softmax
from regawa.functional import num_graphs
from torch.nn.functional import leaky_relu


class GraphAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int):
        super().__init__()  # type: ignore
        self.source = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.target = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.edge = nn.Linear(in_channels, out_channels * heads, bias=False)
        attn = nn.Parameter(empty(1, heads, out_channels))
        nn.init.xavier_normal_(attn)
        self.attn = attn

    def forward(
        self, x_i: Tensor, x_j: Tensor, edge_attribute: Tensor, receivers: Tensor
    ):
        t = self.target(x_j)
        a = segmented_softmax(
            (
                (leaky_relu(self.source(x_i) + t + self.edge(edge_attribute)))
                * self.attn
            ).sum(dim=-1),
            receivers,
            num_graphs(receivers),
        )
        return scatter(a * t, receivers, dim=0, reduce="sum")
