from enum import Enum

import torch as th
from torch import Tensor, nn

from gnn_policy.functional import (
    eval_action_then_node,  # type: ignore
    masked_entropy,  # type: ignore
    node_probs,  # type: ignore
    sample_action_then_node,  # type: ignore
    sample_node,  # type: ignore
)


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


class TwoActionGNNPolicy(nn.Module):
    def __init__(self, num_actions: int, embedding_dim: int, action_mode: ActionMode):
        super().__init__()  # type: ignore

        node_prob_out = {
            ActionMode.ACTION_THEN_NODE: num_actions,
            ActionMode.NODE_THEN_ACTION: 1,
            ActionMode.ACTION_AND_NODE: 1,
        }

        self.node_prob = nn.Linear(
            embedding_dim, node_prob_out[action_mode], bias=False
        )

        self.action_prob = nn.Linear(embedding_dim, num_actions)

        self.num_actions = num_actions

    def action_then_node(
        self, h: Tensor, g: Tensor, batch_idx: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        node_logits = self.node_prob(h).squeeze()
        action_logits = self.action_prob(g)
        node_mask = th.ones(node_logits.shape[0], dtype=th.bool)
        n_g = num_graphs(batch_idx)
        action_mask = th.ones((n_g, self.num_actions), dtype=th.bool)
        return node_logits, action_logits, node_mask, action_mask  # type: ignore

    def node_then_action(
        self, h: Tensor, _: Tensor, batch_idx: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        node_logits = self.node_prob(h).squeeze()
        action_logits = self.action_prob(h)
        node_mask = th.ones(node_logits.shape[0], dtype=th.bool)
        n_g = num_graphs(batch_idx)
        action_mask = th.ones((n_g, self.num_actions), dtype=th.bool)
        return node_logits, action_logits, node_mask, action_mask  # type: ignore

    # differentiable action evaluation
    def forward(
        self, a: Tensor, h: Tensor, g: Tensor, batch_idx: Tensor
    ) -> tuple[Tensor, Tensor]:
        node_logits, action_logits, node_mask, action_mask = self.action_then_node(
            h, g, batch_idx
        )
        logprob, entropy = eval_action_then_node(  # type: ignore
            a,
            action_logits,
            node_logits,
            action_mask,
            node_mask,
            batch_idx,
        )
        return logprob, entropy  # type: ignore

    def sample(
        self, h: Tensor, g: Tensor, batch_idx: Tensor, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        node_logits, action_logits, node_mask, action_mask = self.action_then_node(
            h, g, batch_idx
        )
        actions, logprob, entropy, *_ = sample_action_then_node(  # type: ignore
            action_logits, node_logits, action_mask, node_mask, batch_idx, deterministic
        )
        return actions, logprob, entropy  # type: ignore
