from collections.abc import Callable
from functools import partial
from torch import Tensor, nn

from gnn_policy.functional import (
    eval_action_then_node,  # type: ignore
    node_logits_given_action,  # type: ignore
    sample_action_then_node,  # type: ignore
    segment_sum,  # type: ignore
    masked_entropy,  # type: ignore
)
from regawa.functional import num_graphs, predicate_mask
from regawa.gnn.gnn_classes import SparseTensor
from torch import vmap


def node_mask(action_mask: Tensor) -> Tensor:
    return action_mask[:, 1:].any(1)


def _mul(a, b):
    return a @ b


def value_estimate(
    a: Tensor,
    p_n__a: Tensor,
    q_n__a: Tensor,
    q_a__n: Tensor,
    p_a: Tensor,
    batch_idx: Tensor,
) -> Tensor:
    n_g = max(batch_idx) + 1
    return node_logits_given_action(q_n__a, a[:, 0], batch_idx) @ p_n__a + vmap(_mul)(
        segment_sum(q_a__n, batch_idx, n_g), p_a
    )


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
        self.sample_func = sample_action_then_node
        self.eval_func = eval_action_then_node

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

        actions, logprob, entropy, p_a, p_n__a = x(
            node_logits,
            action_given_node_logits,
            node_given_action_logits,
            mask_actions,
            mask_nodes,
            h.indices,
            n_nodes,
        )

        value = value_estimate(
            actions,
            p_n__a,
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
