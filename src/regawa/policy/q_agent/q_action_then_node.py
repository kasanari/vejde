from functools import partial

from gnn_policy.functional import segment_sum  # type: ignore
from torch import Generator, Tensor, nn

from regawa.data import SparseTensor
from regawa.policy.q_agent.q_value import QValue

from ...nn import linear_reset_parameters
from ..functional import num_graphs


class QActionThenNode(nn.Module):
    def __init__(self, num_actions: int, node_dim: int, rngs: Generator):
        super().__init__()  # type: ignore
        init = partial(linear_reset_parameters, rng=rngs)
        self.q_node__action = init(nn.Linear(node_dim, num_actions))  # Q(n|a)
        self.q_action__node = init(nn.Linear(node_dim, num_actions))  # Q(a|n)

    def forward(
        self,
        h: SparseTensor[Tensor],
    ) -> QValue:
        n_g = num_graphs(h.indices)
        q_n__a = self.q_node__action(h.values)
        q_a__n = self.q_action__node(h.values)
        q_a = segment_sum(q_a__n, index=h.indices, num_segments=n_g)  # type: ignore #TODO this can be done as a weighted sum

        return QValue(q_a, SparseTensor(q_n__a, h.indices))  # type: ignore
