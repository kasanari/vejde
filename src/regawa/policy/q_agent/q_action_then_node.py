from torch import Tensor, nn

from gnn_policy.functional import segment_sum  # type: ignore
from regawa.functional import num_graphs
from regawa.gnn import SparseTensor


class QActionThenNode(nn.Module):
    def __init__(self, num_actions: int, node_dim: int):
        super().__init__()  # type: ignore
        self.q_node__action = nn.Linear(node_dim, num_actions)  # Q(n|a)
        self.q_action__node = nn.Linear(node_dim, num_actions)  # Q(a|n)

    def forward(
        self,
        h: SparseTensor,
    ) -> tuple[Tensor, SparseTensor]:
        n_g = num_graphs(h.indices)

        q_n__a = self.q_node__action(h.values)
        q_a__n = self.q_action__node(h.values)
        q_a = segment_sum(q_a__n, index=h.indices, num_segments=n_g)  # type: ignore #TODO this can be done as a weighted sum

        return q_a, SparseTensor(q_n__a, h.indices)  # type: ignore
