from collections.abc import Callable

import numpy as np
import torch as npl
from numpy.typing import NDArray

from regawa.data import (
    HeteroGraphBuffer,
    TorchHeteroBatchData,
    heterostatedata_to_tensors,
)

from .types import BatchData, UpdateData


def update_step(
    batch_size: int,
    minibatch_size: int,
    update_func: Callable[[TorchHeteroBatchData, BatchData], UpdateData],
    device: str | npl.device,
    rng: np.random.Generator,
):
    def _update_step(
        obs: HeteroGraphBuffer,
        b: BatchData,
        b_inds: NDArray[np.int32],
    ):
        rng.shuffle(b_inds)
        u_datas: list[UpdateData] = []
        stop_training = False
        num_updates = 0  # count number of gradient updates
        for start in range(0, batch_size, minibatch_size):
            mb_inds = b_inds[start : start + minibatch_size]
            u_data = update_func(
                heterostatedata_to_tensors(obs.minibatch(mb_inds), device),
                BatchData(
                    b.actions[mb_inds],
                    b.logprobs[mb_inds],
                    b.advantages[mb_inds],
                    b.returns[mb_inds],
                    b.values[mb_inds],
                    b.rewards[mb_inds],
                    b.dones[mb_inds],
                ),
            )

            u_datas.append(u_data)
            if u_data.stop_training:
                stop_training = True
                break
            num_updates += 1

        return u_datas, stop_training, num_updates

    return _update_step
