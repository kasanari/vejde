from collections.abc import Callable
from functools import partial
from torch import Tensor, nn

from gnn_policy.functional import (
    sample_node_then_action,  # type: ignore
    eval_node_then_action,  # type: ignore
    segment_sum,
    segmented_softmax,  # type: ignore
    softmax,  # type: ignore
    mask_logits,
)
from regawa.functional import (
    node_mask,
    node_then_action_value_estimate,
)
from regawa.gnn.gnn_classes import SparseTensor


PolicyFunc = Callable[
    [Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
]


class NodeThenActionPolicy(nn.Module):
    def __init__(self, num_actions: int, node_dim: int, critic_heads: int = 2):
        super().__init__()  # type: ignore

        self.node_prob = nn.Linear(node_dim, 1, bias=False)
        self.action_given_node_prob = nn.Linear(node_dim, num_actions)

        self.num_actions = num_actions
        self.sample_func = sample_node_then_action  # type: ignore
        self.eval_func = eval_node_then_action  # type: ignore
        self.q_action__node = nn.Linear(node_dim, num_actions * critic_heads)  # Q(a|n)
        self.critic_heads = critic_heads

    def f(
        self, h: SparseTensor, action_mask: Tensor, n_nodes: Tensor, x: PolicyFunc
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        node_logits = self.node_prob(h.values).squeeze()  # ~ln(p(n))
        action_given_node_logits = self.action_given_node_prob(h.values)  # ~ln(p(a|n))
        n_g = n_nodes.shape[0]

        mask_nodes = node_mask(action_mask)
        actions, logprob, entropy, _, p_n = x(  # type: ignore
            action_given_node_logits,
            node_logits,
            mask_nodes,
            action_mask,
            h.indices,
            n_nodes,
        )

        q = self.q_action__node(h.values)
        q = q.view(-1, self.critic_heads, self.num_actions)
        q = q.mean(dim=1)

        p_a__n = softmax(mask_logits(action_given_node_logits, action_mask))  # type: ignore

        # action then node
        value = node_then_action_value_estimate(
            p_a__n,  # type: ignore
            q,
            p_n,  # type: ignore
            partial(segment_sum, index=h.indices, num_segments=n_g),  # type: ignore
        )
        return actions, logprob, entropy, value, p_n, p_a__n  # type: ignore

    # differentiable action evaluation
    def forward(
        self,
        a: Tensor,
        h: SparseTensor,
        action_mask: Tensor,
        n_nodes: Tensor,
    ):
        def p_func(*args):  # type: ignore
            return a, *self.eval_func(a, *args)  # type: ignore

        return self.f(h, action_mask, n_nodes, p_func)[1:]  # type: ignore

    def sample(
        self,
        h: SparseTensor,
        n_nodes: Tensor,
        action_mask: Tensor,
        deterministic: bool = False,
    ):
        p_func = partial(self.sample_func, deterministic=deterministic)  # type: ignore
        return self.f(h, action_mask, n_nodes, p_func)  # type: ignore

    def value(
        self,
        h: SparseTensor,
        n_nodes: Tensor,
        action_mask: Tensor,
    ) -> Tensor:
        m_n = node_mask(action_mask)
        n_g = n_nodes.shape[0]
        node_logits = self.node_prob(h.values).squeeze()  # ~ln(p(n))
        action_given_node_logits = self.action_given_node_prob(h.values)
        p_n = segmented_softmax(mask_logits(node_logits, m_n), h.indices, n_g)
        p_a__n = softmax(mask_logits(action_given_node_logits, action_mask))  # type: ignore

        q = self.q_action__node(h.values)
        q = q.view(-1, self.critic_heads, self.num_actions)
        q = q.mean(dim=1)

        return node_then_action_value_estimate(
            p_a__n,  # type: ignore
            q,
            p_n,  # type: ignore
            partial(segment_sum, index=h.indices, num_segments=n_g),  # type: ignore
        )
