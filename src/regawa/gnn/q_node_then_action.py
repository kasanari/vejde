from torch import nn

from regawa.gnn.gnn_classes import SparseTensor


class QNodeThenAction(nn.Module):
    def __init__(self, num_actions: int, node_dim: int):
        super().__init__()  # type: ignore
        self.q_node = nn.Linear(node_dim, 1)  # Q(n)
        self.q_action__node = nn.Linear(node_dim, num_actions)  # Q(a|n)

    def forward(
        self,
        h: SparseTensor,
    ) -> tuple[SparseTensor, SparseTensor]:
        q_n = self.q_node(h.values)
        q_a__n = self.q_action__node(h.values)
        return SparseTensor(q_n, h.indices), SparseTensor(q_a__n, h.indices)  # type: ignore
