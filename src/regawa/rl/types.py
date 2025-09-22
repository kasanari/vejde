from regawa.data import HeteroGraphBuffer, ObsData


from torch import Tensor

import numpy as np
from typing import NamedTuple, TypeVar

V = TypeVar("V", np.float32, np.bool_, np.int64)


class RolloutData(NamedTuple):
    obs: HeteroGraphBuffer
    last_obs: dict[str, list[ObsData[V]]]
    last_done: Tensor
    global_step: int
    returns: list[float]
    lengths: list[int]


class BatchData(NamedTuple):
    actions: Tensor
    logprobs: Tensor
    advantages: Tensor
    returns: Tensor
    values: Tensor
    rewards: Tensor
    dones: Tensor


class UpdateData(NamedTuple):
    loss: Tensor
    pg_loss: Tensor
    v_loss: Tensor
    entropy_loss: Tensor
    old_approx_kl: Tensor
    approx_kl: Tensor
    grad_norm: Tensor
    clipfrac: float
    stop_training: bool


class PPOParams(NamedTuple):
    clip_coef: float
    norm_adv: bool
    clip_range_vf: float | None
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    target_kl: float | None


class IterationCarry(NamedTuple):
    b: BatchData
    next_obs: dict[str, list[ObsData]]
    next_done: Tensor
    global_step: int
    low_ema: Tensor | None = None
    high_ema: Tensor | None = None
