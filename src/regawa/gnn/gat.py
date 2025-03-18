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

    def alpha(
        self,
        receivers: Tensor,
        senders: Tensor,
        sender_idx: Tensor,
        edge_attribute: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        c = self.out_channels
        h = self.heads
        t = self.target(receivers).view(-1, h, c)
        s = self.source(senders).view(-1, h, c)
        e = self.edge(edge_attribute).view(-1, h, c)
        z = ((leaky_relu(s + t + e)) * self.attn).sum(dim=-1)
        return (
            segment_softmax(
                z,
                sender_idx,
                num_graphs(sender_idx),
                dim=0,
            ).squeeze(0),
            t,
            s,
        )

    def forward(
        self,
        receivers: Tensor,
        senders: Tensor,
        sender_idx: Tensor,
        edge_attribute: Tensor,
    ):
        a, t = self.alpha(receivers, senders, sender_idx, edge_attribute)
        m = t * a.unsqueeze(-1)
        aggr_m = scatter(m, receivers, dim=0, reduce="sum")
        aggr_m = aggr_m.mean(dim=1)  # average over heads
        m = m.mean(dim=1)  # average over heads
        return aggr_m, m
