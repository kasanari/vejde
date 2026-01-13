from typing import Any, NamedTuple, TypeVar

import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from torch import Tensor, as_tensor, device

from regawa.data import (
    HeteroBatchData,
    HeteroObsData,
    heterostatedata_from_obslist,
    n_actions,
)

V = TypeVar("V", np.float32, np.bool_, np.int64)


class ReplayBufferSamples(NamedTuple):
    observations: HeteroBatchData
    next_observations: HeteroBatchData
    actions: Tensor
    dones: Tensor
    rewards: Tensor


def get_single_env(
    obs: dict[str, dict[str, tuple[Any, ...]]], i: int
) -> dict[str, HeteroObsData]:
    return obs[i]


def get_by_type(t: str, buffer):
    for b in buffer:
        yield b[t]


class ReplayBuffer:
    observations: list[tuple[HeteroObsData]]
    next_observations: list[tuple[HeteroObsData]]
    actions: npt.NDArray[np.int32]
    rewards: npt.NDArray[np.float32]
    dones: npt.NDArray[np.float32]

    def __init__(
        self,
        buffer_size: int,
        action_space: spaces.MultiDiscrete,
        device: device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        seed: int | None = None,
    ):
        self.buffer_size = buffer_size

        self.action_dim = n_actions(action_space)
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs
        self.rng = (
            np.random.default_rng() if seed is None else np.random.default_rng(seed)
        )

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        self.optimize_memory_usage = optimize_memory_usage

        self.observations = [None] * self.buffer_size

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = [None] * self.buffer_size

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(
        self,
        obs: tuple[HeteroObsData],
        next_obs: tuple[HeteroObsData],
        action: npt.NDArray[np.int32],
        reward: npt.NDArray[np.float32],
        done: npt.NDArray[np.float32],
        infos: list[dict[str, Any]],
    ) -> None:
        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = obs

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = next_obs
        else:
            self.next_observations[self.pos] = next_obs

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        if not self.optimize_memory_usage:
            upper_bound = self.buffer_size if self.full else self.pos
            batch_inds = self.rng.integers(0, upper_bound, size=batch_size)
            return self._get_samples(batch_inds)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (
                self.rng.integers(1, self.buffer_size, size=batch_size) + self.pos
            ) % self.buffer_size
        else:
            batch_inds = self.rng.integers(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = self.rng.integers(0, high=self.n_envs, size=(len(batch_inds),))

        next_obs = (
            self.observations[(batch_inds + 1) % self.buffer_size][env_indices]
            if self.optimize_memory_usage
            else heterostatedata_from_obslist(
                [self.next_observations[b][e] for b, e in zip(batch_inds, env_indices, strict=False)]
            )
        )

        data = ReplayBufferSamples(
            heterostatedata_from_obslist(
                [self.observations[b][e] for b, e in zip(batch_inds, env_indices, strict=False)]
            ),
            next_obs,
            as_tensor(self.actions[batch_inds, env_indices, :], device=self.device),
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            as_tensor(
                self.dones[batch_inds, env_indices].reshape(-1, 1), device=self.device
            ),
            as_tensor(
                self.rewards[batch_inds, env_indices].reshape(-1, 1), device=self.device
            ),
        )
        return data
