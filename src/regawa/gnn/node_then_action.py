from collections.abc import Callable
from functools import partial
from torch import Tensor, nn

from gnn_policy.functional import (
    sample_node_then_action,  # type: ignore
    eval_node_then_action,  # type: ignore
    segment_sum,
    softmax,  # type: ignore
)
from regawa.functional import node_mask, num_graphs, predicate_mask
from regawa.gnn.gnn_classes import SparseTensor


def value_estimate(
    p_a__n: Tensor,
    q_a__n: Tensor,
    p_n: Tensor,
    q_n: Tensor,
    batch_idx: Tensor,
) -> Tensor:
    n_g = num_graphs(batch_idx)
    segsum = partial(segment_sum, index=batch_idx, num_segments=n_g)
    # Estimate value as the sum of the Q-values of the actions weighted by the probability of the actions
    # V(N) =  Σ_n p(n) * Q(n) + Σ_(a,n) p(a|n) * Q(a|n)
    return segsum(q_n * p_n) + segsum((q_a__n * p_a__n).sum(1))


PolicyFunc = Callable[
    [Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
]


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

    def f(
        self, h: SparseTensor, action_mask: Tensor, n_nodes: Tensor, x: PolicyFunc
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        node_logits = self.node_prob(h.values).squeeze()  # ~ln(p(n))
        action_given_node_logits = self.action_given_node_prob(h.values)  # ~ln(p(a|n))

        mask_actions = predicate_mask(action_mask, h.indices)
        mask_nodes = node_mask(action_mask)
        actions, logprob, entropy, _, p_n = x(  # type: ignore
            action_given_node_logits,
            node_logits,
            mask_actions,
            mask_nodes,
            h.indices,
            n_nodes,
        )

        p_a__n = softmax(action_given_node_logits)

        # action then node
        value = value_estimate(
            p_a__n,
            self.q_action__node(h.values),
            p_n,
            self.q_node(h.values).squeeze(),
            h.indices,
        )
        return actions, logprob, entropy, value  # type: ignore

    # differentiable action evaluation
    def forward(
        self,
        a: Tensor,
        h: SparseTensor,
        action_mask: Tensor,
        n_nodes: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        def p_func(*args):
            return a, *self.eval_func(a, *args)

        return self.f(h, action_mask, n_nodes, p_func)[1:]

    def sample(
        self,
        h: SparseTensor,
        n_nodes: Tensor,
        action_mask: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        p_func = partial(self.sample_func, deterministic=deterministic)
        return self.f(h, action_mask, n_nodes, p_func)  # type: ignore
