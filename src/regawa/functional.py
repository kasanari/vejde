from torch import FloatTensor, Tensor

from gnn_policy.functional import segment_sum
from functools import partial
from regawa.data.torch import SparseTensor


def node_mask(action_mask: Tensor) -> Tensor:
    return action_mask.any(1)


def predicate_mask(action_mask: Tensor, index: Tensor, num_graphs: int) -> Tensor:
    return segment_sum(action_mask, index, num_graphs) > 0


def num_graphs(batch_idx: Tensor) -> int:
    return int(batch_idx.max().item() + 1)


ACTION_DIM = 1


def action_then_node_value_estimate(
    p_n__a: SparseTensor[FloatTensor],  # p(n|a)
    q_n__a: SparseTensor[FloatTensor],  # Q(n|a)
    p_a: Tensor,  # p(a)
    num_graphs: int,
) -> Tensor:
    # Estimate value as the sum of the Q-values of the actions weighted by the probability of the actions
    # V(N) =  Σ_a p(a) Σ_(n) p(n|a) * Q(n|a)
    segsum = partial(segment_sum, index=p_n__a.indices, num_segments=num_graphs)
    return (p_a * segsum(q_n__a.values * p_n__a.values)).sum(ACTION_DIM) 


def node_then_action_value_estimate(
    p_a__n: SparseTensor[FloatTensor],  # p(a|n)
    q_a__n: SparseTensor[FloatTensor],  # Q(a|n)
    p_n: Tensor,  # p(n)
    num_graphs: int,
) -> Tensor:
    # Estimate value as the sum of the Q-values of the actions weighted by the probability of the actions
    # V(N) =  Σ_n p(n) Σ_(a) p(a|n) * Q(a|n)
    segsum = partial(segment_sum, index=p_a__n.indices, num_segments=num_graphs)
    return segsum(p_n * (q_a__n.values * p_a__n.values).sum(ACTION_DIM))  


def action_and_node_value_estimate(
    p_a: Tensor,  # p(a)
    q_a: Tensor,  # Q(a)
    p_n: SparseTensor[FloatTensor],  # p(n)
    q_n: SparseTensor[FloatTensor],  # Q(n)
    num_graphs: int,
) -> Tensor:
    segsum = partial(segment_sum, index=p_n.indices, num_segments=num_graphs)
    return (q_a * p_a).sum(ACTION_DIM) + segsum(q_n.values * p_n.values)
