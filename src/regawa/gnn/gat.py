from torch import Tensor, empty, nn
from torch.nn.functional import leaky_relu
from torch_scatter import scatter  # type: ignore

from gnn_policy.functional import segment_softmax
from regawa.functional import num_graphs


class GraphAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int):
        super().__init__()  # type: ignore
        self.source = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.target = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.edge = nn.Linear(in_channels, out_channels * heads, bias=False)
        attn = nn.Parameter(empty(1, heads, out_channels))
        nn.init.xavier_normal_(attn)
        self.attn = attn
        self.heads = heads
        self.out_channels = out_channels

    def forward(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_attribute: Tensor,
        senders: Tensor,
        receivers: Tensor,
    ):
        c = self.out_channels
        h = self.heads
        t = self.target(x_j).view(-1, h, c)
        s = self.source(x_i).view(-1, h, c)
        e = self.edge(edge_attribute).view(-1, h, c)
        z = ((leaky_relu(s + t + e)) * self.attn).sum(axis=-1)
        a = segment_softmax(
            z,
            senders,
            num_graphs(senders),
            axis=0,
        ).squeeze(0)
        m = t * a.unsqueeze(-1)
        aggr_m = scatter(m, receivers, axis=0, reduce="sum")
        aggr_m = aggr_m.mean(axis=1)
        m = m.mean(axis=1)
        return aggr_m, m
