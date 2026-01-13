import logging

import torch.nn as nn
from torch import Generator as Rngs
from torch import zeros

from regawa.data import TorchFactorGraph

from .acg_factorgraph_layer import FactorGraphLayer
from .attentional_aggregation import AttentionalAggregation

logger = logging.getLogger(__name__)

render_logger = logging.getLogger("message_pass_render")


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
        fg: TorchFactorGraph,
    ) -> TorchFactorGraph:
        # Handle global nodes
        n_g = fg.n_factor.shape[0]
        g = zeros(n_g, self.hidden_size, device=fg.factors.values.device)
        g = self.pre_aggr(fg.globals, n_g) if fg.globals.values.shape[0] > 0 else g

        # add global values to factors
        factors = fg.factors.replace_val(fg.factors.values + g[fg.factors.indices])

        fg = fg._replace(factors=factors)

        logger.debug("Factor Graph")
        logger.debug("Factors:\n%s", fg.factors)
        # logger.debug("Variables:\n%s", variables)
        logger.debug("Edge Index:\n%s", (fg.v_to_f, fg.f_to_v))
        logger.debug("----\n")

        for i, conv in enumerate(self.convs):
            logger.debug("Layer %d", i)
            (variables, factors) = conv(fg)
            fg = fg._replace(
                variables=fg.variables.replace_val(variables),
                factors=fg.factors.replace_val(factors),
            )


        logger.debug("Global Node\n%s", g)
        logger.debug("Message Passing Done\n")
        logger.debug("----\n")

        return fg
