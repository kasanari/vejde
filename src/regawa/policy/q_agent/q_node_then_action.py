from functools import partial

from torch import Generator, nn

from regawa.data import SparseTensor
from regawa.nn import linear_reset_parameters

from .q_value import QValue


class QNodeThenAction(nn.Module):
    def __init__(self, num_actions: int, node_dim: int, rngs: Generator):
        super().__init__()  # type: ignore
        init = partial(linear_reset_parameters, rng=rngs)
        self.q_node = init(nn.Linear(node_dim, 1))  # Q(n)
        self.q_action__node = init(nn.Linear(node_dim, num_actions))  # Q(a|n)

    def forward(
        self,
        h: SparseTensor,
    ) -> QValue:
        q_n = self.q_node(h.values)
        q_a__n = self.q_action__node(h.values)
        return QValue(SparseTensor(q_n, h.indices), SparseTensor(q_a__n, h.indices))
