from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import AttentionalAggregation, MessagePassing  # type: ignore
import logging
from .gnn_classes import MLPLayer

logger = logging.getLogger(__name__)


class FactorGraph(NamedTuple):
    variables: Tensor
    factors: Tensor
    edge_index: Tensor  # edges are (var, factor)
    edge_attr: Tensor
    batch_idx: Tensor


class BipartiteGNNConvVariableToFactor(MessagePassing):
    def __init__(
        self,
        aggr: str,
        in_channels: int,
        out_channels: int,
        activation: nn.Module,
    ):
        super().__init__(aggr=aggr, flow="source_to_target")
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
        return self.propagate(  # type: ignore
            fg.edge_index,
            x=(fg.variables, fg.factors),
            edge_attr=fg.edge_attr,
            size=(fg.variables.size(0), fg.factors.size(0)),
        )

    def edge_update(self, x_j: Tensor) -> Tensor:  # type: ignore
        return x_j

    def message(  # type: ignore
        self,
        x_j: Tensor,
        x_i: Tensor,
        # fac2var_messages: Tensor,
    ) -> Tensor:
        # x_j = x_j - fac2var_messages
        messages = self.message_func(torch.concatenate([x_i, x_j], dim=-1))
        logger.debug("V2F Messages\n%s", messages)
        return messages

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:  # type: ignore
        _, x_o = x
        logger.debug("V2F Aggr out:\n%s", aggr_out)
        new = self.combine(torch.concatenate([x_o, aggr_out], dim=-1))
        return new


class BipartiteGNNConvFactorToVariable(MessagePassing):
    def __init__(
        self, aggr: str, in_channels: int, out_channels: int, activation: nn.Module
    ):
        super().__init__(aggr=aggr, flow="source_to_target")
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

        fac2var_edges = fg.edge_index.flip(0)

        return self.propagate(  # type: ignore
            x=(fg.factors, fg.variables),
            edge_index=fac2var_edges,
            # edge_attr=edge_attr,
            size=(
                fg.factors.size(0),
                fg.variables.size(0),
            ),
        )

    def message(  # type: ignore
        self,
        x_j: Tensor,
        x_i: Tensor,
    ) -> Tensor:  # type: ignore
        messages = self.message_func(torch.concatenate([x_i, x_j], dim=-1))
        logger.debug("F2V Messages\n%s", messages)
        return messages

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:  # type: ignore
        _, x_p = x
        logger.debug("F2V Aggr out\n%s", aggr_out)
        new = x_p + self.combine(torch.concatenate([x_p, aggr_out], dim=-1))
        return new


class GlobalNode(nn.Module):
    def __init__(self, emb_size: int, activation: nn.Module):
        super().__init__()  # type: ignore
        self.aggr = AttentionalAggregation(
            nn.Linear(emb_size, 1), nn.Linear(emb_size, emb_size)
        )
        self.linear = MLPLayer(emb_size * 2, emb_size, activation)
        # self.aggr = SumAggregation()

    def forward(self, x: Tensor, g_prev: Tensor, batch_idx: Tensor) -> Tensor:
        g = self.aggr(x, batch_idx)
        return g_prev + self.linear(torch.concatenate([g, g_prev], dim=-1))


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
            fg.edge_index,
            fg.edge_attr,
            fg.batch_idx,
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
        self.hidden_size = embedding_dim

    def forward(
        self,
        fg: FactorGraph,
    ) -> tuple[FactorGraph, Tensor]:
        variables, factors, edge_index, edge_attr, batch_idx = fg
        num_graphs = int(batch_idx.max().item() + 1)
        g = torch.zeros(num_graphs, self.hidden_size).to(factors.device)
        i = 0
        logger.debug("Factor Graph")
        logger.debug("Factors:\n%s", factors)
        logger.debug("Variables:\n%s", variables)
        logger.debug("Edge Index:\n%s", edge_index)
        logger.debug("----\n")

        for conv in self.convs:
            logger.debug("Layer %d", i)
            logger.debug("Combine Function\n%s", conv.var2factor.combine)
            logger.debug("Message Function\n%s", conv.var2factor.message_func)
            (variables, factors) = conv(fg)
            fg = FactorGraph(variables, factors, edge_index, edge_attr, batch_idx)
            i += 1

        g = self.aggr(factors, g, batch_idx)
        logger.debug("Global Node\n%s", g)
        logger.debug("Message Passing Done\n")
        logger.debug("----\n")

        return fg, g
