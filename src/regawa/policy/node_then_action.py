from collections.abc import Callable
from functools import partial

from gnn_policy.functional import (
    eval_node_then_action,
    mask_logits,
    sample_node_then_action,
    segmented_softmax,
    softmax,
)
from torch import FloatTensor, Generator, Tensor, nn

from regawa.data import SparseTensor
from regawa.data.torch import TorchActionMask
from regawa.nn import linear_reset_parameters

from .functional import node_then_action_value_estimate
from .types import PolicyOutput

PolicyFunc = Callable[
    [Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    PolicyOutput,
]


class NodeThenActionPolicy(nn.Module):
    def __init__(
        self, num_actions: int, node_dim: int, rngs: Generator, critic_heads: int = 2
    ):
        super().__init__()  # type: ignore
        init = partial(linear_reset_parameters, rng=rngs)  # type: ignore
        self.node_prob = init(nn.Linear(node_dim, 1, bias=False))
        self.action_given_node_prob = init(nn.Linear(node_dim, num_actions, bias=False))

        self.num_actions = num_actions
        self.sample_func = sample_node_then_action
        self.eval_func = eval_node_then_action
        self.q_action__node = nn.Linear(
            node_dim, num_actions * critic_heads, bias=False
        )  # Q(a|n)
        nn.init.constant_(self.q_action__node.weight, 0.0)
        self.critic_heads = critic_heads
        self.rngs = rngs

    def f(
        self,
        h: SparseTensor[FloatTensor],
        action_masks: TorchActionMask,
        n_nodes: Tensor,
        x: PolicyFunc,
    ) -> PolicyOutput:
        action_given_node_mask = action_masks.action_type_mask
        node_given_action_mask = action_masks.action_arity_mask.logical_and(
            action_masks.action_type_mask
        )
        node_logits = self.node_prob(h.values).squeeze(-1)  # ~ln(p(n))
        action_given_node_logits = h.map(self.action_given_node_prob)  # ~ln(p(a|n))
        n_g = n_nodes.shape[0]

        actions, logprob, entropy, _, p_n = x(  # type: ignore
            action_given_node_logits.values,
            node_logits,
            action_given_node_mask,
            node_given_action_mask,
            h.indices,
            n_nodes,
        )

        def p_a__n_func(x: Tensor) -> Tensor:
            return softmax(mask_logits(x, action_given_node_mask))

        p_a__n = action_given_node_logits.map(p_a__n_func)
        # action then node
        value = node_then_action_value_estimate(
            p_a__n,
            h.map(self.q_func),
            p_n,  # type: ignore
            n_g,
        )
        return PolicyOutput(actions, logprob, entropy, value, p_n, p_a__n)

    def q_func(self, x: Tensor):
        q = self.q_action__node(x)
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
        rng: Generator,
    ):
        p_func = partial(self.sample_func, rng=rng)
        return self.f(h, action_masks, n_nodes, p_func)  # type: ignore

    def value(
        self,
        h: SparseTensor[FloatTensor],
        n_nodes: Tensor,
        action_masks: TorchActionMask,
    ) -> Tensor:
        n_g = n_nodes.shape[0]

        node_logits = self.node_prob(h.values).squeeze(-1)  # ~ln(p(n))
        action_given_node_logits = h.map(self.action_given_node_prob)
        p_n = segmented_softmax(node_logits, h.indices, n_g)

        def p_a__n_func(x: Tensor) -> Tensor:
            return softmax(mask_logits(x, action_masks.action_type_mask))

        p_a__n = action_given_node_logits.map(p_a__n_func)

        return node_then_action_value_estimate(
            p_a__n,
            h.map(self.q_func),
            p_n,
            n_g,
        )
