from collections.abc import Callable
from enum import Enum
from functools import partial

import torch as th
from torch import Tensor, nn

from gnn_policy.functional import (
    eval_action_then_node,  # type: ignore
    masked_entropy,  # type: ignore
    node_probs,  # type: ignore
    sample_action_then_node,  # type: ignore
    sample_node_then_action,
    eval_node_then_action,
    sample_node,
    segment_sum,  # type: ignore
)
from regawa.gnn.gnn_classes import SparseTensor


class SingleActionGNNPolicy(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()  # type: ignore
        self.node_prob = nn.Linear(embedding_dim, 1)

    def forward(
        self, actions: Tensor, h: Tensor, batch_idx: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        node_logits = self.node_prob(h).squeeze()
        node_mask = th.ones(node_logits.shape[0], dtype=th.bool)
        p = node_probs(node_logits, node_mask, batch_idx)  # type: ignore
        num_graphs = batch_idx.max().item() + 1
        entropy = masked_entropy(p, node_mask, num_graphs)  # type: ignore
        logprob = th.log(p[actions])  # type: ignore
        return logprob, entropy  # type: ignore

    def sample(self, h: Tensor, batch_idx: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        node_logits = self.node_prob(h).squeeze()
        node_mask = th.ones(node_logits.shape[0], dtype=th.bool)
        actions, prob, entropy, *_ = sample_node(node_logits, node_mask, batch_idx)  # type: ignore
        logprob = th.log(prob[actions])  # type: ignore
        return actions, logprob, entropy  # type: ignore


class ActionMode(Enum):
    ACTION_THEN_NODE = 0
    NODE_THEN_ACTION = 1
    ACTION_AND_NODE = 2


def num_graphs(batch_idx: Tensor) -> int:
    return int(batch_idx.max().item() + 1)


def node_mask(action_mask: Tensor) -> Tensor:
    return action_mask[:, 1:].any(1)


def predicate_mask(action_mask: Tensor, segsum: Callable[[Tensor], Tensor]) -> Tensor:
    return segsum(action_mask) > 0


class TwoActionGNNPolicy(nn.Module):
    def __init__(
        self, num_actions: int, node_dim: int, graph_dim: int, action_mode: ActionMode
    ):
        super().__init__()  # type: ignore

        node_prob_out = {
            ActionMode.ACTION_THEN_NODE: num_actions,
            ActionMode.NODE_THEN_ACTION: 1,
            ActionMode.ACTION_AND_NODE: 1,
        }

        self.node_prob = nn.Linear(node_dim, node_prob_out[action_mode], bias=False)

        action_input_dim = (
            node_dim if action_mode == ActionMode.NODE_THEN_ACTION else graph_dim
        )

        self.action_prob = nn.Linear(action_input_dim, num_actions)

        action_eval_funcs = {
            ActionMode.ACTION_THEN_NODE: eval_action_then_node,
            ActionMode.NODE_THEN_ACTION: eval_node_then_action,
        }
        action_sample_funcs = {
            ActionMode.ACTION_THEN_NODE: sample_action_then_node,
            ActionMode.NODE_THEN_ACTION: sample_node_then_action,
        }

        action_logit_func: dict[ActionMode, Callable[[Tensor, Tensor], Tensor]] = {
            ActionMode.ACTION_THEN_NODE: lambda _, g: self.action_prob(g),
            ActionMode.NODE_THEN_ACTION: lambda h, _: self.action_prob(h),
        }

        self.num_actions = num_actions
        self.sample_func = action_sample_funcs[action_mode]
        self.eval_func = action_eval_funcs[action_mode]
        self.action_prob_func = action_logit_func[action_mode]

    # differentiable action evaluation
    def forward(
        self,
        a: Tensor,
        h: SparseTensor,
        g: Tensor,
        action_mask: Tensor,
        n_nodes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        node_logits = self.node_prob(h.values).squeeze()
        action_logits = self.action_prob_func(h.values, g)
        segsum = partial(
            segment_sum, num_segments=num_graphs(h.indices), index=h.indices
        )
        logprob, entropy = self.eval_func(  # type: ignore
            a,
            action_logits,
            node_logits,
            predicate_mask(action_mask, segsum),
            node_mask(action_mask),
            h.indices,
            n_nodes,
        )
        return logprob, entropy  # type: ignore

    def sample(
        self,
        h: SparseTensor,
        g: Tensor,
        n_nodes: Tensor,
        action_mask: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        node_logits = self.node_prob(h.values).squeeze()
        action_logits = self.action_prob_func(h.values, g)
        segsum = partial(
            segment_sum, num_segments=num_graphs(h.indices), index=h.indices
        )
        actions, logprob, entropy, *_ = self.sample_func(  # type: ignore
            action_logits,
            node_logits,
            predicate_mask(action_mask, segsum),
            node_mask(action_mask),
            h.indices,
            n_nodes,
            deterministic,
        )
        return actions, logprob, entropy  # type: ignore
