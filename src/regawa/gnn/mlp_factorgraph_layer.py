import torch.nn as nn
from torch import Generator as Rngs, concatenate, empty
from torch import Tensor
from torch_scatter import scatter

from regawa.data import FactorGraph
from .mlp import MLPLayer
from .mp_rendering import Lazy, to_graphviz_bp
import logging

render_logger = logging.getLogger("message_pass_render")


def update(x: Tensor, y: Tensor, mlp: nn.Module, edge_attr: Tensor) -> Tensor:
    return mlp(concatenate((x, y, edge_attr), axis=-1))


class MLPFactorGraphLayer(nn.Module):
    def __init__(
        self, embedding_dim: int, aggregation: str, activation: nn.Module, rngs: Rngs
    ):
        super().__init__()  # type: ignore
        self.variable_transform = MLPLayer(
            embedding_dim * 3, embedding_dim, activation, rngs
        )
        self.factor_transform = MLPLayer(
            embedding_dim * 3, embedding_dim, activation, rngs
        )

    def forward(
        self,
        fg: FactorGraph,
        prev_m_f_to_v: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        v_to_f = fg.v_to_f
        f_to_v = fg.f_to_v
        variable = fg.variables.values
        factor = fg.factors.values
        edge_attr = fg.edge_attr

        # aggregate messages to variables
        sum_f_to_v = scatter(
            prev_m_f_to_v, v_to_f, reduce="sum", dim_size=variable.size(0), dim=0
        )

        # do not send messages back to sender
        m_v_to_f = sum_f_to_v[v_to_f] - prev_m_f_to_v

        # combine messages with node value and edge attribute
        # m_v_to_f = (
        #     self.variable_transform(concatenate((m_v_to_f, variable[v_to_f]), axis=-1))
        # ) + edge_attr
        m_v_to_f = update(
            variable[v_to_f], m_v_to_f, self.variable_transform, edge_attr
        )

        # aggregate messages
        sum_v_to_f = scatter(
            m_v_to_f, f_to_v, reduce="sum", dim_size=factor.size(0), dim=0
        )

        # do not send messages back to sender
        m_f_to_v = sum_v_to_f[f_to_v] - m_v_to_f

        # combine messages with node value and edge attribute
        m_f_to_v = update(factor[f_to_v], m_f_to_v, self.factor_transform, edge_attr)

        # update node values as a combination of aggregated messages and node value
        # new_factor = self.factor_combine(concatenate((sum_v_to_f, factor), axis=-1))
        # new_variable = self.variable_combine(
        #     concatenate((sum_f_to_v, variable), axis=-1)
        # )

        # overwrite node values with aggregated messages
        new_factor = sum_v_to_f + factor
        new_variable = sum_f_to_v + variable

        # keep node values the same
        # new_factor = factor
        # new_variable = variable

        # use GIN style update
        # new_factor = gin_update(
        #     factor, sum_v_to_f, self.factor_epsilon, self.factor_combine
        # )
        # new_variable = gin_update(
        #     variable, sum_f_to_v, self.variable_epsilon, self.variable_combine
        # )

        render_logger.debug(
            "%s",
            Lazy(
                lambda: to_graphviz_bp(
                    prev_m_f_to_v,
                    m_v_to_f,
                    sum_v_to_f,
                    sum_f_to_v,
                    f_to_v,
                    v_to_f,
                )
            ),
        )

        return m_f_to_v, new_factor, new_variable
