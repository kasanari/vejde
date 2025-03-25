import torch
from torch import Tensor


def lambda_returns(
    gamma: float,
    lambda_: float,
):
    @torch.inference_mode()
    def compute_lambda_values(
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
    ) -> Tensor:
        ret = torch.zeros_like(rewards)
        ret[-1] = values[-1]
        for t in reversed(range(len(rewards))[:-1]):
            ret[t] = rewards[t] + gamma * (~(dones[t + 1] > 0)) * (
                (1 - lambda_) * values[t + 1] + lambda_ * ret[t + 1]
            )
        return ret

    return compute_lambda_values


@torch.inference_mode()
def return_scale(
    returns: Tensor, low_ema: Tensor | None, high_ema: Tensor | None, decay: float
) -> tuple[Tensor, Tensor, Tensor]:
    low, high = returns.quantile(0.05), returns.quantile(0.95)
    low_ema = low if low_ema is None else decay * low_ema + (1 - decay) * low  # type: ignore
    high_ema = (
        high if high_ema is None else decay * high_ema + (1 - decay) * high  # type: ignore
    )
    s = high_ema - low_ema
    return s, low_ema, high_ema


def test_lambda_return():
    gamma = 0.99
    lambda_ = 0.95
    compute_lambda_values = lambda_returns(gamma, lambda_)
    rewards = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    values = torch.tensor([2.0, 2.0, 2.0, 2.0, 1.0, 0.0])
    dones = torch.tensor([0, 0, 0, 0, 0, 1])
    last_values = torch.tensor([0.0])
    next_step_is_terminal = False
    adv, ret = compute_lambda_values(
        rewards, values, dones, last_values, next_step_is_terminal
    )
    pass


if __name__ == "__main__":
    test_lambda_return()
