from collections.abc import Callable
from functools import partial

from torch import FloatTensor, Generator as Rngs
from torch import Tensor, nn

from gnn_policy.functional import (
    eval_action_then_node,
    marginalize,
    mask_logits,
    sample_action_then_node,
    segment_softmax,
)
from regawa.data.torch import TorchActionMask
from regawa.functional import (
    action_then_node_value_estimate,
    num_graphs,
)
from regawa.data import SparseTensor
from .types import PolicyOutput

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
        h: SparseTensor[FloatTensor],
        action_masks: TorchActionMask,
        n_nodes: Tensor,
        x: PolicyFunc,
    ):
        node_logits = self.node_prob(h.values).squeeze(-1)  # ~ln(p(n))
        action_given_node_logits = h.map(self.action_given_node_prob)  # ~ln(p(a|n))
        node_given_action_logits = h.map(self.node_given_action_prob)  # ~ln(p(n|a))
        n_g = num_graphs(h.indices)
        action_given_node_mask = action_masks.action_type_mask
        node_given_action_mask = action_masks.action_arity_mask.logical_and(
            action_masks.action_type_mask
        )

        actions, logprob, entropy, p_a, _ = x(
            node_logits,
            action_given_node_logits.values,
            node_given_action_logits.values,
            action_given_node_mask,
            node_given_action_mask,
            h.indices,
            n_nodes,
        )

        def p_n_given_a(x: Tensor):
            return segment_softmax(
                mask_logits(x, node_given_action_mask),
                h.indices,
                n_g,
            )

        p_n__a = node_given_action_logits.map(p_n_given_a)

        value = action_then_node_value_estimate(
            p_n__a,
            h.map(self.q_func),
            p_a,
            n_g,
        )

        return PolicyOutput(actions, logprob, entropy, value, p_a, p_n__a)

    def q_func(self, x: Tensor):
        q = self.q_node__action(x)
        q = q.view(-1, self.critic_heads, self.num_actions)
        return q.mean(axis=1)

    # differentiable action evaluation
    def forward(
        self,
        a: Tensor,
        h: SparseTensor[FloatTensor],
        action_masks: TorchActionMask,
        n_nodes: Tensor,
    ):
        def p_func(*args):  # type: ignore
            return a, *self.eval_func(a, *args)  # type: ignore

        return self.f(h, action_masks, n_nodes, p_func)[1:]  # type: ignore

    def sample(
        self,
        h: SparseTensor[FloatTensor],
        n_nodes: Tensor,
        action_masks: TorchActionMask,
        deterministic: bool = False,
    ):
        p_func = partial(self.sample_func, deterministic=deterministic)  # type: ignore
        return self.f(h, action_masks, n_nodes, p_func)

    def value(
        self,
        h: SparseTensor[FloatTensor],
        n_nodes: Tensor,
        action_masks: TorchActionMask,
    ) -> Tensor:
        node_logits = self.node_prob(h.values).squeeze(-1)  # ~ln(p(n))
        action_given_node_logits = h.map(self.action_given_node_prob)
        node_given_action_logits = h.map(self.node_given_action_prob)

        action_given_node_mask = action_masks.action_type_mask
        node_given_action_mask = action_masks.action_arity_mask.logical_and(
            action_masks.action_type_mask
        )

        n_g = n_nodes.shape[0]
        p_a = marginalize(
            node_logits,
            mask_logits(action_given_node_logits.values, action_given_node_mask),
            h.indices,
            n_g,
        )

        def p_n_given_a(x: Tensor):
            return segment_softmax(
                mask_logits(x, node_given_action_mask),
                h.indices,
                n_g,
            )

        p_n__a = node_given_action_logits.map(p_n_given_a)

        return action_then_node_value_estimate(
            p_n__a,
            h.map(self.q_func),
            p_a,
            n_g,
        )
