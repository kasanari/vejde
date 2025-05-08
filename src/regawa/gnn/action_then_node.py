from collections.abc import Callable
from functools import partial

from torch import Generator as Rngs
from torch import Tensor, nn

from gnn_policy.functional import (
    eval_action_then_node,
    marginalize,
    mask_logits,
    sample_action_then_node,
    segment_softmax,
    segment_sum,
)
from regawa.functional import (
    action_then_node_value_estimate,
    num_graphs,
    predicate_mask,
)
from regawa.gnn.gnn_classes import SparseTensor

PolicyFunc = Callable[
    [Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
]


class ActionThenNodePolicy(nn.Module):
    def __init__(
        self, num_actions: int, node_dim: int, rngs: Rngs, critic_heads: int = 2
    ):
        super().__init__()  # type: ignore

        self.node_prob = nn.Linear(node_dim, 1, bias=False)
        self.action_given_node_prob = nn.Linear(node_dim, num_actions, bias=False)
        self.node_given_action_prob = nn.Linear(node_dim, num_actions, bias=False)

        self.num_actions = num_actions
        self.sample_func = sample_action_then_node  # type: ignore
        self.eval_func = eval_action_then_node  # type: ignore

        self.q_node__action = nn.Linear(
            node_dim, num_actions * critic_heads, bias=False
        )  # Q(n|a)
        self.critic_heads = critic_heads

        nn.init.constant_(self.q_node__action.weight, 0.0)

    def f(
        self,
        h: SparseTensor,
        action_type_mask: Tensor,
        action_arity_mask: Tensor,
        n_nodes: Tensor,
        x: PolicyFunc,
    ):
        node_logits = self.node_prob(h.values).squeeze(-1)  # ~ln(p(n))
        action_given_node_logits = self.action_given_node_prob(h.values)  # ~ln(p(a|n))
        node_given_action_logits = self.node_given_action_prob(h.values)  # ~ln(p(n|a))
        n_g = num_graphs(h.indices)
        action_given_node_mask = action_type_mask
        node_given_action_mask = action_arity_mask.logical_and(action_type_mask)

        actions, logprob, entropy, p_a, _ = x(
            node_logits,
            action_given_node_logits,
            node_given_action_logits,
            action_given_node_mask,
            node_given_action_mask,
            h.indices,
            n_nodes,
        )

        p_n__a = segment_softmax(  # type: ignore
            mask_logits(node_given_action_logits, node_given_action_mask),
            h.indices,
            n_g,
        )
        segsum = partial(segment_sum, index=h.indices, num_segments=n_g)  # type: ignore

        q = self.q_node__action(h.values)
        q = q.view(-1, self.critic_heads, self.num_actions)
        q = q.mean(axis=1)

        value = action_then_node_value_estimate(
            p_n__a,  # type: ignore
            q,
            p_a,
            segsum,  # type: ignore
        )

        return actions, logprob, entropy, value, p_a, p_n__a  # type: ignore

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
        return self.f(h, action_type_mask, action_arity_mask, n_nodes, p_func)

    def value(
        self,
        h: SparseTensor,
        n_nodes: Tensor,
        action_type_mask: Tensor,
        action_arity_mask: Tensor,
    ) -> Tensor:
        node_logits = self.node_prob(h.values).squeeze(-1)  # ~ln(p(n))
        action_given_node_logits = self.action_given_node_prob(h.values)
        node_given_action_logits = self.node_given_action_prob(h.values)

        action_given_node_mask = action_type_mask
        node_given_action_mask = action_arity_mask.logical_and(action_type_mask)

        n_g = n_nodes.shape[0]
        p_a = marginalize(
            node_logits,
            mask_logits(action_given_node_logits, action_given_node_mask),
            h.indices,
            n_g,
        )

        p_n__a = segment_softmax(  # type: ignore
            mask_logits(node_given_action_logits, node_given_action_mask),
            h.indices,
            n_g,
        )
        segsum = partial(segment_sum, index=h.indices, num_segments=n_g)  # type: ignore

        q = self.q_node__action(h.values)
        q = q.view(-1, self.critic_heads, self.num_actions)
        q = q.mean(axis=1)

        return action_then_node_value_estimate(
            p_n__a,  # type: ignore
            q,
            p_a,
            segsum,  # type: ignore
        )
