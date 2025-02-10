from collections.abc import Callable
from torch import Tensor

from gnn_policy.functional import segment_sum


def node_mask(action_mask: Tensor) -> Tensor:
    return action_mask.any(1)


def predicate_mask(action_mask: Tensor, index: Tensor, num_graphs: int) -> Tensor:
    return segment_sum(action_mask, index, num_graphs) > 0


def num_graphs(batch_idx: Tensor) -> int:
    return int(batch_idx.max().item() + 1)


def action_then_node_value_estimate(
    p_n__a: Tensor,  # p(n|a)
    q_n__a: Tensor,  # Q(n|a)
    p_a: Tensor,  # p(a)
    segsum: Callable[[Tensor], Tensor],
) -> Tensor:
    # Estimate value as the sum of the Q-values of the actions weighted by the probability of the actions

    # V(N) =  Σ_a p(a) Σ_(n) p(n|a) * Q(n|a)
    return (p_a * segsum(q_n__a * p_n__a)).sum(-1)  # type: ignore


def node_then_action_value_estimate(
    p_a__n: Tensor,  # p(a|n)
    q_a__n: Tensor,  # Q(a|n)
    p_n: Tensor,  # p(n)
    segsum: Callable[[Tensor], Tensor],
) -> Tensor:
    # Estimate value as the sum of the Q-values of the actions weighted by the probability of the actions
    # V(N) =  Σ_n p(n) Σ_(a) p(a|n) * Q(a|n)
    return segsum(p_n * (q_a__n * p_a__n).sum(1))  # type: ignore


def action_and_node_value_estimate(
    p_a: Tensor,  # p(a)
    q_a: Tensor,  # Q(a)
    p_n: Tensor,  # p(n)
    q_n: Tensor,  # Q(n)
    segsum: Callable[[Tensor], Tensor],
) -> Tensor:
    return (q_a * p_a).sum(1) + segsum(q_n * p_n)
