import logging
from collections.abc import Callable

import numpy as np
import torch as npl
from gymnasium.vector import VectorEnv
from numpy.typing import NDArray
from torch import Tensor

from regawa.data import (
    HeteroGraphBuffer,
    HeteroIndexedFactorGraph,
    heterostatedata_to_tensors,
)
from regawa.data.batch.batch_func import heterobatch

from . import lambda_return
from .agent import Agent
from .explained_var import explained_variance
from .symexp import symlog
from .types import BatchData, IterationCarry, RolloutData, UpdateData

logger = logging.getLogger(__name__)


def iteration_step(
    anneal_lr: bool,
    agent: Agent,
    batch_size: int,
    update_epochs: int,
    envs: VectorEnv[HeteroIndexedFactorGraph, NDArray[np.int32], NDArray[np.int32]],
    optimizer: npl.optim.Optimizer,
    learning_rate: float,
    num_iterations: int,
    rollout_func: Callable[
        [dict[str, list[HeteroIndexedFactorGraph]], Tensor, BatchData, int],
        tuple[RolloutData, BatchData],
    ],
    gae_func: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor]],
    update_func: Callable[
        [HeteroGraphBuffer, BatchData, NDArray[np.int32]],
        tuple[list[UpdateData], bool, int],
    ],
    lambda_return_func: Callable[[Tensor, Tensor, Tensor], Tensor],
    ema_decay: float,
    device: str | npl.device,
):
    def _iteration_step(
        iteration: int,
        carry: IterationCarry,
    ):
        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        r_data, b = rollout_func(
            carry.next_obs,
            carry.next_done,
            carry.b,
            carry.global_step,
        )

        # bootstrap value if not done
        with npl.no_grad():
            next_obs_batch = heterostatedata_to_tensors(
                heterobatch(r_data.last_obs), device
            )
            advantages, returns = gae_func(
                b.rewards,
                b.dones,
                b.values,
                agent.get_value(next_obs_batch).reshape(1, -1),
                r_data.last_done,
            )

        # flatten the batch
        # b_obs = Batch.from_data_list(obs)

        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = b.values.reshape(-1)

        lambda_r = lambda_return_func(b.rewards, b.values, b.dones)
        s, low_ema, high_ema = lambda_return.return_scale(
            lambda_r, carry.low_ema, carry.high_ema, ema_decay
        )
        b_advantages = b_advantages / max(1.0, s.item())

        # plot histogram of returns
        # import matplotlib.pyplot as plt

        # plt.hist(b_returns.cpu().numpy(), bins=100)
        # plt.savefig("returns.png")
        # plt.close()

        symlogged_b_returns = symlog(b_returns)
        symlogged_b_values = symlog(b_values)

        # plt.hist(b_returns.cpu().numpy(), bins=100)
        # plt.savefig("returns_symlog.png")
        # plt.close()

        flattened_b = BatchData(
            b.actions.reshape((-1, *envs.single_action_space.shape)),  # type: ignore
            b.logprobs.reshape(-1),
            b_advantages,
            symlogged_b_returns,
            symlogged_b_values,
            b.rewards.reshape(-1),
            b.dones.reshape(-1),
        )

        # Optimizing the policy and value network

        b_inds = np.arange(batch_size)
        u_datas: list[UpdateData] = []

        total_num_updates = carry.num_updates

        for epoch in range(update_epochs):
            u_data, stop_training, num_updates = update_func(
                r_data.obs, flattened_b, b_inds
            )
            u_datas.extend(u_data)
            total_num_updates += num_updates
            if stop_training:
                logger.info(
                    f"Early stopping at step {epoch} due to reaching max kl: {u_datas[-1].loss.approx_kl:.2f}"
                )

        carry = IterationCarry(
            BatchData(
                b.actions,
                b.logprobs,
                b_advantages,
                b_returns,
                b.values,
                b.rewards,
                b.dones,
            ),
            r_data.last_obs,
            r_data.last_done,
            r_data.global_step,
            total_num_updates,
            low_ema,
            high_ema,
        )
        return (r_data, u_datas, explained_variance(b_values, b_returns), s, carry)

    return _iteration_step
