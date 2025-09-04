import logging

import torch.nn as nn
from torch import Generator as Rngs
from torch import Tensor, concatenate, zeros_like

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

        v_to_f = fg.v_to_f
        f_to_v = fg.f_to_v
        variables = fg.variables.values
        factors = fg.factors.values
        edge_attr = fg.edge_attr

        x_i = factors[f_to_v]  # map factors to messages
        x_j = variables[v_to_f]  # map variables to messages

        aggr_m, m = self.mp(x_i, x_j, f_to_v, edge_attr, num_segments=factors.shape[0])

        # combine target and aggregated message
        # also, check for empty aggr_m
        x = (factors, aggr_m) if aggr_m.shape[0] > 0 else (factors, zeros_like(factors))
        x = concatenate(x, axis=-1)
        x = self.combine(x)

        render_logger.debug(
            "%s", Lazy(lambda: to_graphviz(m, v_to_f, f_to_v, variables, factors))
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

        v_to_f = fg.v_to_f
        f_to_v = fg.f_to_v
        variables = fg.variables.values
        factors = fg.factors.values
        edge_attr = fg.edge_attr

        x_i = variables[v_to_f]  # map variables to messages
        x_j = factors[f_to_v]  # map factors to messages

        aggr_m, m = self.mp(
            x_i, x_j, v_to_f, zeros_like(edge_attr), num_segments=variables.shape[0]
        )

        # combine target and aggregated message
        x = (variables, aggr_m)
        x = concatenate(x, axis=-1)
        x = self.combine(x)

        # skip connection. I am not sure why this helps, but it does.
        x = variables + x
        render_logger.debug(
            "%s",
            Lazy(
                lambda: to_graphviz(m, v_to_f, f_to_v, variables, factors, reverse=True)
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
        )

        n_h_v = self.factor2var(
            new_fg,
        )

        logger.debug("New Variable\n%s", n_h_v)

        return n_h_v, n_h_f
