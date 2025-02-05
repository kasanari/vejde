from collections.abc import Callable
from functools import partial
from torch import Tensor, nn

from gnn_policy.functional import (
    sample_node_then_action,  # type: ignore
    eval_node_then_action,  # type: ignore
    segment_sum,  # type: ignore
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
    def __init__(self, num_actions: int, node_dim: int):
        super().__init__()  # type: ignore

        self.node_prob = nn.Linear(node_dim, 1, bias=False)
        self.action_given_node_prob = nn.Linear(node_dim, num_actions)

        self.num_actions = num_actions
        self.sample_func = sample_node_then_action  # type: ignore
        self.eval_func = eval_node_then_action  # type: ignore

        self.q_node = nn.Linear(node_dim, 1)  # Q(n)
        self.q_action__node = nn.Linear(node_dim, num_actions)  # Q(a|n)

    def f(
        self, h: SparseTensor, action_mask: Tensor, n_nodes: Tensor, x: PolicyFunc
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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

        p_a__n = softmax(mask_logits(action_given_node_logits, action_mask))  # type: ignore

        # action then node
        value = node_then_action_value_estimate(
            p_a__n,  # type: ignore
            self.q_action__node(h.values),
            p_n,  # type: ignore
            self.q_node(h.values).squeeze(),
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
