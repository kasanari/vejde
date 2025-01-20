from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor
import logging
from torch import max
from torch_scatter import scatter

# from torch import scatter
from .gnn_classes import MLPLayer

logger = logging.getLogger(__name__)


def segmented_softmax(src: Tensor, index: Tensor, dim: int = 0) -> Tensor:
    n_nodes = max(index).int() + 1
    src_max = scatter(src.detach(), index, dim, dim_size=n_nodes, reduce="max")
    out = src - src_max.index_select(dim, index)
    out = out.exp()
    out_sum = scatter(out, index, dim, dim_size=n_nodes, reduce="sum") + 1e-16
    out_sum = out_sum.index_select(dim, index)
    return out / out_sum


class BatchIdx(NamedTuple):
    global_: Tensor
    factor: Tensor
    variable: Tensor


class FactorGraph(NamedTuple):
    variables: Tensor
    factors: Tensor
    senders: Tensor
    receivers: Tensor
    edge_attr: Tensor
    n_factor: Tensor
    globals_: Tensor
    batch: BatchIdx


class BipartiteGNNConvVariableToFactor(nn.Module):
    def __init__(
        self,
        aggr: str,
        in_channels: int,
        out_channels: int,
        activation: nn.Module,
    ):
        super().__init__()
        self.combine = MLPLayer(out_channels * 2, out_channels, activation)
        self.message_func = MLPLayer(in_channels * 2, out_channels, activation)

    def forward(
        self,
        fg: FactorGraph,
    ):
        """
        x_p: Tensor of shape (num_predicates, embedding_dim),
        x_o: Tensor of shape (num_objects, embedding_dim),
        edge_index: Tensor of shape (2, num_edges),
        edge_attr: Tensor of shape (num_edges,)
        """

        senders = fg.senders
        receivers = fg.receivers
        variables = fg.variables
        factors = fg.factors

        x_i = factors[receivers]
        x_j = variables[senders]
        x = (x_i, x_j)
        logger.debug("V2F X_j\n%s", x_j)
        logger.debug("V2F X_i\n%s", x_i)
        x = torch.concatenate(x, dim=-1)
        x = self.message_func(x)
        logger.debug("V2F Messages\n%s", x)
        x = scatter(x, receivers, dim=0, reduce="sum")
        logger.debug("V2F Aggr out:\n%s", x)
        x = (factors, x)
        x = torch.concatenate(x, dim=-1)
        x = self.combine(x)
        return x


class BipartiteGNNConvFactorToVariable(nn.Module):
    def __init__(
        self, aggr: str, in_channels: int, out_channels: int, activation: nn.Module
    ):
        super().__init__()
        # self.root = MLPLayer(in_channels, out_channels, activation)
        self.combine = MLPLayer(out_channels * 2, out_channels, activation)
        self.message_func = MLPLayer(in_channels * 2, out_channels, activation)
        # self.messages = None
        # self.relation_embedding = nn.Embedding(num_relations, in_channels)
        # nn.init.orthogonal_(self.relation_embedding.weight)

    def forward(
        self,
        fg: FactorGraph,
    ):
        """
        x_p: Tensor of shape (num_predicates, embedding_dim),
        x_o: Tensor of shape (num_objects, embedding_dim),
        edge_index: Tensor of shape (2, num_edges),
        edge_attr: Tensor of shape (num_edges,)
        """

        senders = fg.senders
        receivers = fg.receivers
        variables = fg.variables
        factors = fg.factors

        x_i = variables[senders]
        x_j = factors[receivers]
        logger.debug("V2F X_j\n%s", x_j)
        logger.debug("V2F X_i\n%s", x_i)
        x = (x_i, x_j)
        x = torch.concatenate(x, dim=-1)
        x = self.message_func(x)
        logger.debug("F2V Messages\n%s", x)
        x = scatter(x, senders, dim=0, reduce="sum")
        logger.debug("F2V Aggr out\n%s", x)
        x = (variables, x)
        x = torch.concatenate(x, dim=-1)
        x = self.combine(x)
        x = variables + x
        return x


class AttentionalAggregation(nn.Module):
    def __init__(self, emb_size: int, activation: nn.Module):
        super().__init__()  # type: ignore

        self.gate = nn.Linear(emb_size, 1)
        self.attn = nn.Linear(emb_size, emb_size)

    def forward(self, nodes: Tensor, batch_idx: Tensor) -> Tensor:
        x = self.gate(nodes)
        x = segmented_softmax(x, batch_idx)
        x = x * self.attn(nodes)
        x = scatter(x, batch_idx, dim=0, reduce="sum")
        return x


class GlobalNode(nn.Module):
    def __init__(self, emb_size: int, activation: nn.Module):
        super().__init__()  # type: ignore

        self.attn = AttentionalAggregation(emb_size, activation)

        self.linear = MLPLayer(emb_size * 2, emb_size, activation)
        # self.aggr = SumAggregation()

    def forward(self, nodes: Tensor, g_prev: Tensor, batch_idx: Tensor) -> Tensor:
        x = self.attn(nodes, batch_idx)
        x = (x, g_prev)
        x = torch.concatenate(x, dim=-1)
        x = g_prev + self.linear(x)
        return x


class FactorGraphLayer(nn.Module):
    def __init__(self, embedding_dim: int, aggregation: str, activation: nn.Module):
        super().__init__()  # type: ignore
        self.var2factor = BipartiteGNNConvVariableToFactor(
            aggregation, embedding_dim, embedding_dim, activation
        )

        self.factor2var = BipartiteGNNConvFactorToVariable(
            aggregation, embedding_dim, embedding_dim, activation
        )

    def forward(
        self,
        fg: FactorGraph,
    ) -> tuple[Tensor, Tensor]:
        n_h_f = self.var2factor(
            fg,
        )

        logger.debug("New Factor\n%s", n_h_f)

        new_fg = FactorGraph(
            fg.variables,
            n_h_f,
            fg.senders,
            fg.receivers,
            fg.edge_attr,
            fg.n_factor,
            fg.globals_,
            fg.batch,
        )

        n_h_v = self.factor2var(
            new_fg,
        )

        logger.debug("New Variable\n%s", n_h_v)

        return n_h_v, n_h_f


class BipartiteGNN(nn.Module):
    def __init__(
        self,
        layers: int,
        embedding_dim: int,
        aggregation: str,
        activation: nn.Module,
    ):
        super().__init__()  # type: ignore

        self.convs = nn.ModuleList(
            [
                FactorGraphLayer(embedding_dim, aggregation, activation)
                for _ in range(layers)
            ]
        )

        # self.aggrs = [GlobalNode(embedding_dim, activation) for _ in range(layers)]
        self.aggr = GlobalNode(embedding_dim, activation)
        self.pre_aggr = AttentionalAggregation(embedding_dim, activation)
        self.hidden_size = embedding_dim

    def forward(
        self,
        fg: FactorGraph,
    ) -> tuple[FactorGraph, Tensor]:
        num_graphs = int(fg.batch.factor.max().item() + 1)
        g = torch.zeros(num_graphs, self.hidden_size).to(fg.factors.device)
        g = (
            self.pre_aggr(fg.globals_, fg.batch.global_)
            if fg.globals_.shape[0] > 0
            else g
        )

        variables = fg.variables + g[fg.batch.variable]

        i = 0
        logger.debug("Factor Graph")
        logger.debug("Factors:\n%s", fg.factors)
        logger.debug("Variables:\n%s", variables)
        logger.debug("Edge Index:\n%s", (fg.senders, fg.receivers))
        logger.debug("----\n")

        for conv in self.convs:
            logger.debug("Layer %d", i)
            logger.debug("Combine Function\n%s", conv.var2factor.combine)
            logger.debug("Message Function\n%s", conv.var2factor.message_func)
            (variables, factors) = conv(fg)
            fg = FactorGraph(
                variables,
                factors,
                fg.senders,
                fg.receivers,
                fg.edge_attr,
                fg.n_factor,
                fg.globals_,
                fg.batch,
            )
            i += 1

        g = self.aggr(fg.factors, g, fg.batch.factor)
        logger.debug("Global Node\n%s", g)
        logger.debug("Message Passing Done\n")
        logger.debug("----\n")

        return fg, g
