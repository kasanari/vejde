from collections.abc import Callable
from functools import partial

from torch import Generator as Rngs
from torch import Tensor, nn

from gnn_policy.functional import eval_node_then_action  # type: ignore
from gnn_policy.functional import sample_node_then_action  # type: ignore
from gnn_policy.functional import segmented_softmax  # type: ignore
from gnn_policy.functional import softmax  # type: ignore
from gnn_policy.functional import mask_logits, segment_sum
from regawa.functional import node_then_action_value_estimate
from regawa.data import SparseTensor

PolicyFunc = Callable[
    [Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
]


class NodeThenActionPolicy(nn.Module):
    def __init__(
        self, num_actions: int, node_dim: int, rngs: Rngs, critic_heads: int = 2
    ):
        super().__init__()  # type: ignore

        self.node_prob = nn.Linear(node_dim, 1, bias=False)
        self.action_given_node_prob = nn.Linear(node_dim, num_actions, bias=False)

        self.num_actions = num_actions
        self.sample_func = sample_node_then_action  # type: ignore
        self.eval_func = eval_node_then_action  # type: ignore
        self.q_action__node = nn.Linear(
            node_dim, num_actions * critic_heads, bias=False
        )  # Q(a|n)
        self.critic_heads = critic_heads

    def f(
        self,
        h: SparseTensor,
        action_type_mask: Tensor,
        action_arity_mask: Tensor,
        n_nodes: Tensor,
        x: PolicyFunc,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        action_given_node_mask = action_type_mask
        node_given_action_mask = action_arity_mask.logical_and(action_type_mask)
        node_logits = self.node_prob(h.values).squeeze(-1)  # ~ln(p(n))
        action_given_node_logits = self.action_given_node_prob(h.values)  # ~ln(p(a|n))
        n_g = n_nodes.shape[0]

        actions, logprob, entropy, _, p_n = x(  # type: ignore
            action_given_node_logits,
            node_logits,
            action_given_node_mask,
            node_given_action_mask,
            h.indices,
            n_nodes,
        )

        q = self.q_action__node(h.values)
        q = q.view(-1, self.critic_heads, self.num_actions)
        q = q.mean(axis=1)

        p_a__n = softmax(mask_logits(action_given_node_logits, action_given_node_mask))  # type: ignore

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
        action_type_mask: Tensor,
        action_arity_mask: Tensor,
        n_nodes: Tensor,
    ):
        def p_func(*args):  # type: ignore
            return a, *self.eval_func(a, *args)  # type: ignore

        return self.f(h, action_type_mask, action_arity_mask, n_nodes, p_func)[1:]  # type: ignore

    def sample(
        self,
        h: SparseTensor,
        n_nodes: Tensor,
        action_type_mask: Tensor,
        action_arity_mask: Tensor,
        deterministic: bool = False,
    ):
        p_func = partial(self.sample_func, deterministic=deterministic)  # type: ignore
        return self.f(h, action_type_mask, action_arity_mask, n_nodes, p_func)  # type: ignore

    def value(
        self,
        h: SparseTensor,
        n_nodes: Tensor,
        action_type_mask: Tensor,
        action_arity_mask: Tensor,
    ) -> Tensor:
        n_g = n_nodes.shape[0]

        node_logits = self.node_prob(h.values).squeeze(-1)  # ~ln(p(n))
        action_given_node_logits = self.action_given_node_prob(h.values)
        p_n = segmented_softmax(node_logits, h.indices, n_g)
        p_a__n = softmax(mask_logits(action_given_node_logits, action_type_mask))  # type: ignore

        q = self.q_action__node(h.values)
        q = q.view(-1, self.critic_heads, self.num_actions)
        q = q.mean(axis=1)

        return node_then_action_value_estimate(
            p_a__n,  # type: ignore
            q,
            p_n,  # type: ignore
            partial(segment_sum, index=h.indices, num_segments=n_g),  # type: ignore
        )
