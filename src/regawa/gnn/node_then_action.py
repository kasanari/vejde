from torch import Tensor, nn

from gnn_policy.functional import (
    sample_node_then_action,  # type: ignore
    eval_node_then_action,  # type: ignore
    segment_sum,  # type: ignore
)
from regawa.functional import num_graphs
from regawa.gnn.gnn_classes import SparseTensor


def node_mask(action_mask: Tensor) -> Tensor:
    return action_mask[:, 1:].any(1)


def value_estimate(
    a: Tensor,
    p_a__n: Tensor,
    q_a__n: Tensor,
    p_n: Tensor,
    q_n: Tensor,
    batch_idx: Tensor,
) -> Tensor:
    n_g = num_graphs(batch_idx)
    return (q_a__n[a[:, 1]] * p_a__n).sum(1) + segment_sum(q_n * p_n, batch_idx, n_g)


class NodeThenActionPolicy(nn.Module):
    def __init__(self, num_actions: int, node_dim: int):
        super().__init__()  # type: ignore

        self.node_prob = nn.Linear(node_dim, 1, bias=False)
        self.action_given_node_prob = nn.Linear(node_dim, num_actions)
        self.node_given_action_prob = nn.Linear(node_dim, num_actions)

        self.num_actions = num_actions
        self.sample_func = sample_node_then_action
        self.eval_func = eval_node_then_action

        self.q_node = nn.Linear(node_dim, 1)  # Q(n)
        self.q_action__node = nn.Linear(node_dim, num_actions)  # Q(a|n)

    # differentiable action evaluation
    def forward(
        self,
        a: Tensor,
        h: SparseTensor,
        action_mask: Tensor,
        n_nodes: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        node_logits = self.node_prob(h.values).squeeze()  # ~ln(p(n))
        action_given_node_logits = self.action_given_node_prob(h.values)  # ~ln(p(a|n))

        logprob, entropy, p_a__n, p_n = self.eval_func(  # type: ignore
            a,
            action_given_node_logits,
            node_logits,
            action_mask,
            node_mask(action_mask),
            h.indices,
            n_nodes,
        )

        # action then node
        value = value_estimate(
            a,
            p_a__n,
            self.q_action__node(h.values),
            p_n,
            self.q_node(h.values).squeeze(),
            h.indices,
        )

        return logprob, entropy, value  # type: ignore

    def sample(
        self,
        h: SparseTensor,
        n_nodes: Tensor,
        action_mask: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        node_logits = self.node_prob(h.values).squeeze()  # ~ln(p(n))
        action_given_node_logits = self.action_given_node_prob(h.values)  # ~ln(p(a|n))

        actions, logprob, entropy, p_a__n, p_n = self.sample_func(  # type: ignore
            action_given_node_logits,
            node_logits,
            action_mask,
            node_mask(action_mask),
            h.indices,
            n_nodes,
            deterministic,
        )

        # action then node
        value = value_estimate(
            actions,
            p_a__n,
            self.q_action__node(h.values),
            p_n,
            self.q_node(h.values).squeeze(),
            h.indices,
        )

        # node then action
        # value = self.node_q(h) * p_n + self.node_action_q(h)[a[:, 1]] * p_a__n

        return actions, logprob, entropy, value  # type: ignore
