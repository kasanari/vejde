from functools import partial

from gnn_policy.functional import segment_sum
from torch import FloatTensor, Tensor

from regawa.data.torch import SparseTensor
from regawa.functional import ACTION_DIM


def sac_node_then_action_value_estimate(
    p_a__n: SparseTensor[FloatTensor],  # p(a|n)
    q_a__n: SparseTensor[FloatTensor],  # Q(a|n)
    p_n: SparseTensor[FloatTensor],  # p(n)
    logp_a__n: SparseTensor[FloatTensor],  # log p(a|n)
    logp_n: SparseTensor[FloatTensor],  # log p(n)
    alpha: float,  # entropy coefficient
    n_graphs: int,
) -> Tensor:
    # Estimate value as the sum of the Q-values of the actions weighted by the probability of the actions
    # No Q value for first action since the intermediate state has no reward
    # V(N) =  Σ_n p(n) [- α log p(n) + Σ_a p(a|n) [Q(a|n) - α log p(a|n)] ]
    segsum = partial(segment_sum, index=p_n.indices, num_segments=n_graphs)
    return segsum(
        p_n.values
        * (
            # (-alpha * logp_n.values) +
            (q_a__n.values - alpha * logp_a__n.values) * p_a__n.values
        ).sum(ACTION_DIM)
    )  # type: ignore


def sac_node_then_action_policy_loss(
    p_a__n: SparseTensor[FloatTensor],  # p(a|n)
    q_a__n: SparseTensor[FloatTensor],  # Q(a|n)
    p_n: SparseTensor[FloatTensor],  # p(n)
    logp_a__n: SparseTensor[FloatTensor],  # log p(a|n)
    logp_n: SparseTensor[FloatTensor],  # log p(n)
    alpha: float,  # entropy coefficient
    n_graphs: int,
) -> Tensor:
    # Policy loss for node-then-action policy
    # J_π = E_n~π [ α log p(n) - Σ_a p(a|n) [Q(a|n) - α log p(a|n)] ]
    segsum = partial(segment_sum, index=p_n.indices, num_segments=n_graphs)
    return segsum(
        p_n.values
        * (
            # (alpha * logp_n.values) +
            (alpha * logp_a__n.values - q_a__n.values) * p_a__n.values
        ).sum(ACTION_DIM)
    ).mean()  # type: ignore


def sac_action_then_node_policy_loss(
    p_n__a: SparseTensor[FloatTensor],  # p(n|a)
    q_n__a: SparseTensor[FloatTensor],  # Q(n|a)
    p_a: Tensor,  # p(a)
    q_a: Tensor,  # Q(a)
    logp_a: Tensor,  # log p(a)
    logp_n__a: SparseTensor[FloatTensor],  # log p(n|a)
    alpha: float,  # entropy coefficient
    num_graphs: int,
) -> Tensor:
    # Policy loss for action-then-node policy
    # J_π = E_a~π [ α log p(a) - Σ_n p(n|a) [Q(n|a) - α log p(n|a)] ]
    segsum = partial(segment_sum, index=p_n__a.indices, num_segments=num_graphs)
    return (
        (
            p_a
            * (
                # (alpha * logp_a - q_a) +
                segsum((alpha * logp_n__a.values - q_n__a.values) * p_n__a.values)
            )
        )
        .sum(ACTION_DIM)
        .mean()
    )  # type: ignore


def sac_action_then_node_value_estimate(
    p_n__a: SparseTensor[FloatTensor],  # p(n|a)
    q_n__a: SparseTensor[FloatTensor],  # Q(n|a)
    p_a: Tensor,  # p(a)
    q_a: Tensor,  # Q(a)
    logp_a: Tensor,  # log p(a)
    logp_n__a: SparseTensor[FloatTensor],  # log p(n|a)
    alpha: float,  # entropy coefficient
    n_graphs: int,
) -> Tensor:
    # Estimate value as the sum of the Q-values of the actions weighted by the probability of the actions
    # No Q value for first action since the intermediate state has no reward
    # V(N) =  Σ_a p(a) [- α log p(a) + Σ_n p(n|a) [Q(n|a) - α log p(n|a)] ]
    segsum = partial(segment_sum, index=p_n__a.indices, num_segments=n_graphs)
    return (
        p_a
        * (
            # (q_a - alpha * logp_a) +
            segsum((q_n__a.values - alpha * logp_n__a.values) * p_n__a.values)
        )
    ).sum(ACTION_DIM)  # type: ignore


def sac_action_then_node_entropy(
    p_n__a: SparseTensor[FloatTensor],  # p(n|a)
    p_a: Tensor,  # p(a)
    logp_a: Tensor,  # log p(a)
    logp_n__a: SparseTensor[FloatTensor],  # log p(n|a)
    alpha: Tensor,  # entropy coefficient
    entropy_target_a: Tensor,
    entropy_target_n__a: Tensor,
    n_graphs: int,
) -> Tensor:
    # Entropy loss for action-then-node policy
    #
    segsum = partial(segment_sum, index=p_n__a.indices, num_segments=n_graphs)
    entropy = (
        p_a.detach()
        * (
            # -alpha * (logp_a + entropy_target_a).detach() +
            segsum(
                -alpha.exp()
                * (logp_n__a.values + entropy_target_n__a).detach()
                * p_n__a.values.detach()
            )
        )
    ).sum(ACTION_DIM)  # type: ignore
    return entropy.mean()
