import logging
from collections.abc import Callable

import torch.nn as nn
from torch import Generator as Rngs
from torch import Tensor, concatenate, zeros, zeros_like
from torch.linalg import norm

from regawa.gnn.attentional_aggregation import AttentionalAggregation
from regawa.gnn.data import FactorGraph
from regawa.gnn.mp_rendering import Lazy, to_graphviz
from regawa.gnn.simple_mp import MessagePass

from .gnn_classes import MLPLayer, SparseTensor

logger = logging.getLogger(__name__)

render_logger = logging.getLogger("message_pass_render")


class BipartiteGNNConvVariableToFactor(nn.Module):
    def __init__(
        self,
        aggr: str,
        in_channels: int,
        out_channels: int,
        activation: nn.Module,
        rngs: Rngs,
    ):
        super().__init__()  # type: ignore
        self.combine = MLPLayer(out_channels * 2, out_channels, activation, rngs)
        self.mp = MessagePass(in_channels, out_channels, aggr, activation, rngs)
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

        senders = fg.v_to_f
        receivers = fg.f_to_v
        variables = fg.variables.values
        factors = fg.factors.values
        edge_attr = fg.edge_attr

        x_i = factors[receivers]
        x_j = variables[senders]

        aggr_m, m = self.mp(x_i, x_j, receivers, edge_attr)

        x = (factors, aggr_m) if aggr_m.shape[0] > 0 else (factors, zeros_like(factors))
        x = concatenate(x, axis=-1)
        x = self.combine(x)
        render_logger.debug(
            "%s", Lazy(lambda: to_graphviz(m, senders, receivers, variables, factors))
        )
        return x


class BipartiteGNNConvFactorToVariable(nn.Module):
    def __init__(
        self,
        aggr: str,
        in_channels: int,
        out_channels: int,
        activation: nn.Module,
        rngs: Rngs,
    ):
        super().__init__()  # type: ignore

        self.combine = MLPLayer(out_channels * 2, out_channels, activation, rngs)
        self.mp = MessagePass(in_channels, out_channels, aggr, activation, rngs)

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

        senders = fg.v_to_f
        receivers = fg.f_to_v
        variables = fg.variables.values
        factors = fg.factors.values
        edge_attr = fg.edge_attr

        x_i = variables[senders]
        x_j = factors[receivers]

        aggr_m, m = self.mp(x_i, x_j, senders, zeros_like(edge_attr))
        x = (variables, aggr_m)
        x = concatenate(x, axis=-1)
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
    def __init__(
        self, embedding_dim: int, aggregation: str, activation: nn.Module, rngs: Rngs
    ):
        super().__init__()  # type: ignore
        self.var2factor = BipartiteGNNConvVariableToFactor(
            aggregation, embedding_dim, embedding_dim, activation, rngs
        )

        self.factor2var = BipartiteGNNConvFactorToVariable(
            aggregation, embedding_dim, embedding_dim, activation, rngs
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
            fg.globals,
            fg.v_to_f,
            fg.f_to_v,
            fg.edge_attr,
            fg.n_variable,
            fg.n_factor,
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
        rngs: Rngs,
    ):
        super().__init__()  # type: ignore

        def f(i: int):
            logger.info("Initing Layer %d\n", i)
            return FactorGraphLayer(embedding_dim, aggregation, activation, rngs)

        self.convs = nn.ModuleList([f(i) for i in range(layers)])
        self.pre_aggr = AttentionalAggregation(embedding_dim)
        self.hidden_size = embedding_dim

    def forward(
        self,
        fg: FactorGraph,
    ) -> FactorGraph:
        n_g = fg.n_factor.shape[0]
        g = zeros(n_g, self.hidden_size, device=fg.factors.values.device)
        g = self.pre_aggr(fg.globals, n_g) if fg.globals.values.shape[0] > 0 else g

        factors = SparseTensor(
            fg.factors.values + g[fg.factors.indices], fg.factors.indices
        )
        fg = FactorGraph(
            fg.variables,
            factors,
            fg.globals,
            fg.v_to_f,
            fg.f_to_v,
            fg.edge_attr,
            fg.n_variable,
            fg.n_factor,
            fg.action_mask,
        )

        i = 0
        logger.debug("Factor Graph")
        logger.debug("Factors:\n%s", fg.factors)
        # logger.debug("Variables:\n%s", variables)
        logger.debug("Edge Index:\n%s", (fg.v_to_f, fg.f_to_v))
        logger.debug("----\n")

        for conv in self.convs:
            logger.debug("Layer %d", i)
            (variables, factors) = conv(fg)
            fg = FactorGraph(
                SparseTensor(variables, fg.variables.indices),
                SparseTensor(factors, fg.factors.indices),
                fg.globals,
                fg.v_to_f,
                fg.f_to_v,
                fg.edge_attr,
                fg.n_variable,
                fg.n_factor,
                fg.action_mask,
            )
            i += 1

        logger.debug("Global Node\n%s", g)
        logger.debug("Message Passing Done\n")
        logger.debug("----\n")

        return fg
