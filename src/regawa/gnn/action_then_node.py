from collections.abc import Callable
from functools import partial
from torch import Tensor, nn

from gnn_policy.functional import (
    eval_action_then_node,  # type: ignore
    sample_action_then_node,  # type: ignore,
    segment_softmax,  # type: ignore
    segment_sum,  # type: ignore
)
from regawa.functional import node_mask, num_graphs, predicate_mask
from regawa.gnn.gnn_classes import SparseTensor


def value_estimate(
    p_n__a: Tensor,
    q_n__a: Tensor,
    q_a__n: Tensor,
    p_a: Tensor,
    batch_idx: Tensor,
) -> Tensor:
    # Estimate value as the sum of the Q-values of the actions weighted by the probability of the actions
    # we assume Q(a) = Σ_n Q(a | n)
    n_g = num_graphs(batch_idx)
    segsum = partial(segment_sum, index=batch_idx, num_segments=n_g)  # type: ignore
    q_a = segsum(q_a__n)  # type: ignore
    # V(N) =  Σ_a p(a) * Q(a) + Σ_(a,n) p(n|a) * Q(n|a)
    return (q_a * p_a).sum(1) + segsum((q_n__a * p_n__a).sum(-1))  # type: ignore


PolicyFunc = Callable[
    [Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
]


class ActionThenNodePolicy(nn.Module):
    def __init__(self, num_actions: int, node_dim: int):
        super().__init__()  # type: ignore

        self.node_prob = nn.Linear(node_dim, 1, bias=False)
        self.action_given_node_prob = nn.Linear(node_dim, num_actions)
        self.node_given_action_prob = nn.Linear(node_dim, num_actions)

        self.num_actions = num_actions
        self.sample_func = sample_action_then_node  # type: ignore
        self.eval_func = eval_action_then_node  # type: ignore

        self.q_node = nn.Linear(node_dim, 1)  # Q(n)
        self.q_node__action = nn.Linear(node_dim, num_actions)  # Q(n|a)
        self.q_action__node = nn.Linear(node_dim, num_actions)  # Q(a|n)

    def f(self, h: SparseTensor, action_mask: Tensor, n_nodes: Tensor, x: PolicyFunc):
        node_logits = self.node_prob(h.values).squeeze()  # ~ln(p(n))
        action_given_node_logits = self.action_given_node_prob(h.values)  # ~ln(p(a|n))
        node_given_action_logits = self.node_given_action_prob(h.values)  # ~ln(p(n|a))
        mask_actions = predicate_mask(
            action_mask, h.indices
        )  # TODO do not use predicate_mask
        mask_nodes = node_mask(action_mask)

        actions, logprob, entropy, p_a, _ = x(
            node_logits,
            action_given_node_logits,
            node_given_action_logits,
            mask_actions,
            mask_nodes,
            h.indices,
            n_nodes,
        )

        p_n__a = segment_softmax(  # type: ignore
            node_given_action_logits, h.indices, num_graphs(h.indices)
        )

        value = value_estimate(
            p_n__a,  # type: ignore
            self.q_node__action(h.values),
            self.q_action__node(h.values),
            p_a,
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
        def p_func(*args):  # type: ignore
            return a, *self.eval_func(a, *args)  # type: ignore

        return self.f(h, action_mask, n_nodes, p_func)[1:]  # type: ignore

    def sample(
        self,
        h: SparseTensor,
        n_nodes: Tensor,
        action_mask: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        p_func = partial(self.sample_func, deterministic=deterministic)  # type: ignore
        return self.f(h, action_mask, n_nodes, p_func)  # type: ignore
