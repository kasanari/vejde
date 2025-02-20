import torch as th
from torch import Tensor, nn

from gnn_policy.functional import masked_entropy  # type: ignore
from gnn_policy.functional import node_probs  # type: ignore
from gnn_policy.functional import sample_node  # type: ignore
from regawa.functional import num_graphs


class SingleActionGNNPolicy(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()  # type: ignore
        self.node_prob = nn.Linear(embedding_dim, 1)

    def forward(
        self, actions: Tensor, h: Tensor, batch_idx: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        node_logits = self.node_prob(h).squeeze()
        p = node_probs(node_logits, node_mask, batch_idx)  # type: ignore
        n_g = num_graphs(batch_idx)
        entropy = masked_entropy(p, node_mask, n_g)  # type: ignore
        logprob = th.log(p[actions])  # type: ignore
        return logprob, entropy  # type: ignore

    def sample(self, h: Tensor, batch_idx: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        node_logits = self.node_prob(h).squeeze()
        actions, prob, entropy, *_ = sample_node(node_logits, node_mask, batch_idx)  # type: ignore
        logprob = th.log(prob[actions])  # type: ignore
        return actions, logprob, entropy  # type: ignore
