from functools import partial

from gnn_policy.functional import (
    masked_entropy,  # type: ignore
    node_probs,  # type: ignore
    sample_node,  # type: ignore
)
from torch import Generator, Tensor, log, nn

from regawa.nn import linear_reset_parameters

from .functional import num_graphs


class SingleActionGNNPolicy(nn.Module):
    def __init__(self, embedding_dim: int, rng: Generator):
        super().__init__()  # type: ignore
        init = partial(linear_reset_parameters, rng=rng)
        self.node_prob = init(nn.Linear(embedding_dim, 1))

    def forward(
        self, actions: Tensor, h: Tensor, batch_idx: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        node_logits = self.node_prob(h).squeeze(-1)
        p = node_probs(node_logits, batch_idx)  # type: ignore
        n_g = num_graphs(batch_idx)
        entropy = masked_entropy(p, n_g)  # type: ignore
        logprob = log(p[actions])  # type: ignore
        return logprob, entropy  # type: ignore

    def sample(self, h: Tensor, batch_idx: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        node_logits = self.node_prob(h).squeeze(-1)
        actions, prob, entropy, *_ = sample_node(node_logits, batch_idx)  # type: ignore
        logprob = log(prob[actions])  # type: ignore
        return actions, logprob, entropy  # type: ignore
