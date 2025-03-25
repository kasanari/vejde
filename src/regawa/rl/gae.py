import torch
from torch import Tensor

# numpy-like
npl = torch


def gae(
    num_steps: int,
    gamma: float,
    gae_lambda: float,
    device: npl.device | str,
):
    @npl.inference_mode()
    def _gae(
        rewards: Tensor,
        dones: Tensor,
        values: Tensor,
        next_value: Tensor,
        next_step_is_terminal: Tensor,  # env will autoreset on next call to step
    ) -> tuple[Tensor, Tensor]:
        advantages = npl.zeros_like(rewards).to(device)
        last_gae_lam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_is_not_terminal = 1.0 - next_step_is_terminal.float()
                next_values = next_value
            else:
                next_is_not_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]
            delta = rewards[t] + gamma * next_values * next_is_not_terminal - values[t]
            last_gae_lam = (
                delta + gamma * gae_lambda * next_is_not_terminal * last_gae_lam
            )
            advantages[t] = last_gae_lam
        returns = advantages + values  # negates the -values[t] to get td targets
        return advantages, returns

    return _gae
