import logging
from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from regawa.gnn.attentional_aggregation import AttentionalAggregation
from regawa.gnn.simple_mp import MessagePass

from .gnn_classes import MLPLayer, SparseTensor

logger = logging.getLogger(__name__)

render_logger = logging.getLogger("message_pass_render")


class FactorGraph(NamedTuple):
    variables: SparseTensor
    factors: SparseTensor
    senders: Tensor
    receivers: Tensor
    edge_attr: Tensor
    n_factor: Tensor
    globals_: SparseTensor
    action_mask: Tensor


class Lazy:
    def __init__(self, func: Callable[[], str]):
        self.func = func

    def __str__(self) -> str:
        return self.func()


def to_graphviz(
    messages: Tensor,
    senders: Tensor,
    receivers: Tensor,
    variables: Tensor,
    factors: Tensor,
    reverse: bool = False,
):
    output = "digraph G {"

    norm = lambda x: torch.linalg.norm(x, ord=2, axis=-1)
    ro = lambda x: torch.round(x, decimals=2)

    v_norm = norm(variables)
    f_norm = norm(factors)
    m_norm = norm(messages)

    edge_string = (
        lambda v, f: f'"{v}_v" -> "{f}_f"' if not reverse else f'"{f}_f" -> "{v}_v"'
    )

    for i, v in enumerate(v_norm):
        output += f'"{i}_v" [label="{v:0.2f}", shape=ellipse, color=blue];'

    for i, f in enumerate(f_norm):
        output += f'"{i}_f" [label="{f:0.2f}", shape=rectangle, color=red];'

    for s, r, m in zip(senders, receivers, m_norm):
        output += f'{edge_string(s, r)} [label="{m:0.2f}"];'

    output += "}"
    return output


class BipartiteGNNConvVariableToFactor(nn.Module):
    def __init__(
        self,
        aggr: str,
        in_channels: int,
        out_channels: int,
        activation: nn.Module,
    ):
        super().__init__()  # type: ignore
        self.combine = MLPLayer(out_channels * 2, out_channels, activation)
        self.mp = MessagePass(in_channels, out_channels, aggr, activation)
        self.aggr = aggr

        logger.info("Variable to Factor\n")
        logger.info("Combine Function\n%s", self.combine)

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
        variables = fg.variables.values
        factors = fg.factors.values

        x_i = factors[receivers]
        x_j = variables[senders]

        aggr_m, m = self.mp(x_i, x_j, receivers)

        x = (
            (factors, aggr_m)
            if aggr_m.shape[0] > 0
            else (factors, torch.zeros_like(factors))
        )
        x = torch.concatenate(x, axis=-1)
        x = self.combine(x)
        render_logger.debug(
            "%s", Lazy(lambda: to_graphviz(m, senders, receivers, variables, factors))
        )
        return x


class BipartiteGNNConvFactorToVariable(nn.Module):
    def __init__(
        self, aggr: str, in_channels: int, out_channels: int, activation: nn.Module
    ):
        super().__init__()  # type: ignore

        self.combine = MLPLayer(out_channels * 2, out_channels, activation)
        self.mp = MessagePass(in_channels, out_channels, aggr, activation)

        logger.info("Factor to Variable\n")
        logger.info("Combine Function\n%s", self.combine)

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
        variables = fg.variables.values
        factors = fg.factors.values

        x_i = variables[senders]
        x_j = factors[receivers]

        aggr_m, m = self.mp(x_i, x_j, senders)
        x = (variables, aggr_m)
        x = torch.concatenate(x, axis=-1)
        x = self.combine(x)
        x = variables + x
        render_logger.debug(
            "%s",
            Lazy(
                lambda: to_graphviz(
                    m, senders, receivers, variables, factors, reverse=True
                )
            ),
        )
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
            SparseTensor(n_h_f, fg.factors.indices),
            fg.senders,
            fg.receivers,
            fg.edge_attr,
            fg.n_factor,
            fg.globals_,
            fg.action_mask,
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

        def f(i: int):
            logger.info("Initing Layer %d\n", i)
            return FactorGraphLayer(embedding_dim, aggregation, activation)

        self.convs = nn.ModuleList([f(i) for i in range(layers)])
        self.pre_aggr = AttentionalAggregation(embedding_dim)
        self.hidden_size = embedding_dim

    def forward(
        self,
        fg: FactorGraph,
    ) -> FactorGraph:
        n_g = fg.n_factor.shape[0]
        g = torch.zeros(n_g, self.hidden_size).to(fg.factors.values.device)
        g = self.pre_aggr(fg.globals_, n_g) if fg.globals_.values.shape[0] > 0 else g

        factors = SparseTensor(
            fg.factors.values + g[fg.factors.indices], fg.factors.indices
        )
        fg = FactorGraph(
            fg.variables,
            factors,
            fg.senders,
            fg.receivers,
            fg.edge_attr,
            fg.n_factor,
            fg.globals_,
            fg.action_mask,
        )

        i = 0
        logger.debug("Factor Graph")
        logger.debug("Factors:\n%s", fg.factors)
        # logger.debug("Variables:\n%s", variables)
        logger.debug("Edge Index:\n%s", (fg.senders, fg.receivers))
        logger.debug("----\n")

        for conv in self.convs:
            logger.debug("Layer %d", i)
            (variables, factors) = conv(fg)
            fg = FactorGraph(
                SparseTensor(variables, fg.variables.indices),
                SparseTensor(factors, fg.factors.indices),
                fg.senders,
                fg.receivers,
                fg.edge_attr,
                fg.n_factor,
                fg.globals_,
                fg.action_mask,
            )
            i += 1

        logger.debug("Global Node\n%s", g)
        logger.debug("Message Passing Done\n")
        logger.debug("----\n")

        return fg
