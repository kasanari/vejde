from regawa.functional import ACTION_DIM


from torch import Tensor


from collections.abc import Callable


def sac_action_then_node_policy_loss(
    p_n__a: Tensor,  # p(n|a)
    q_n__a: Tensor,  # Q(n|a)
    p_a: Tensor,  # p(a)
    logp_a: Tensor,  # log p(a)
    logp_n__a: Tensor,  # log p(n|a)
    alpha: float,  # entropy coefficient
    segsum: Callable[[Tensor], Tensor],
) -> Tensor:
    # Policy loss for action-then-node policy
    # J_π = E_a~π [ α log p(a) - Σ_n p(n|a) [Q(n|a) - α log p(n|a)] ]
    return (
        (p_a * ((alpha * logp_a) + segsum((alpha * logp_n__a - q_n__a) * p_n__a)))
        .sum(ACTION_DIM)
        .mean()
    )  # type: ignore


def sac_node_then_action_value_estimate(
    p_a__n: Tensor,  # p(a|n)
    q_a__n: Tensor,  # Q(a|n)
    p_n: Tensor,  # p(n)
    logp_a__n: Tensor,  # log p(a|n)
    logp_n: Tensor,  # log p(n)
    alpha: float,  # entropy coefficient
    segsum: Callable[[Tensor], Tensor],
) -> Tensor:
    # Estimate value as the sum of the Q-values of the actions weighted by the probability of the actions
    # No Q value for first action since the intermediate state has no reward
    # V(N) =  Σ_n p(n) [- α log p(n) + Σ_a p(a|n) [Q(a|n) - α log p(a|n)] ]
    return segsum(
        p_n
        * ((-alpha * logp_n) + (q_a__n - alpha * logp_a__n) * p_a__n).sum(ACTION_DIM)
    )  # type: ignore


def sac_node_then_action_policy_loss(
    p_a__n: Tensor,  # p(a|n)
    q_a__n: Tensor,  # Q(a|n)
    p_n: Tensor,  # p(n)
    logp_a__n: Tensor,  # log p(a|n)
    logp_n: Tensor,  # log p(n)
    alpha: float,  # entropy coefficient
    segsum: Callable[[Tensor], Tensor],
) -> Tensor:
    # Policy loss for node-then-action policy
    # J_π = E_n~π [ α log p(n) - Σ_a p(a|n) [Q(a|n) - α log p(a|n)] ]
    return segsum(
        p_n * ((alpha * logp_n) + (alpha * logp_a__n - q_a__n) * p_a__n).sum(ACTION_DIM)
    ).mean()  # type: ignore


def sac_action_then_node_value_estimate(
    p_n__a: Tensor,  # p(n|a)
    q_n__a: Tensor,  # Q(n|a)
    p_a: Tensor,  # p(a)
    logp_a: Tensor,  # log p(a)
    logp_n__a: Tensor,  # log p(n|a)
    alpha: float,  # entropy coefficient
    segsum: Callable[[Tensor], Tensor],
) -> Tensor:
    # Estimate value as the sum of the Q-values of the actions weighted by the probability of the actions
    # No Q value for first action since the intermediate state has no reward
    # V(N) =  Σ_a p(a) [- α log p(a) + Σ_n p(n|a) [Q(n|a) - α log p(n|a)] ]
    return (
        p_a * ((-alpha * logp_a) + segsum((q_n__a - alpha * logp_n__a) * p_n__a))
    ).sum(ACTION_DIM)  # type: ignore


def sac_action_then_node_entropy(
    p_n__a: Tensor,  # p(n|a)
    p_a: Tensor,  # p(a)
    logp_a: Tensor,  # log p(a)
    logp_n__a: Tensor,  # log p(n|a)
    alpha: Tensor,  # entropy coefficient
    entropy_target_a: float,
    entropy_target_n__a: float,
    segsum: Callable[[Tensor], Tensor],
) -> Tensor:
    # Entropy loss for action-then-node policy
    #
    entropy = (
        p_a.detach()
        * (
            -alpha * (logp_a + entropy_target_a).detach()
            + segsum(
                -alpha.exp()
                * (logp_n__a + entropy_target_n__a).detach()
                * p_n__a.detach()
            )
        )
    ).sum(ACTION_DIM)  # type: ignore
    return entropy.mean()
