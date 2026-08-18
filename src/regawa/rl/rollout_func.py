from collections import deque
from collections.abc import Mapping

import numpy as np
import torch as npl
from gymnasium.vector import VectorEnv
from numpy.typing import NDArray
from torch import Generator, Tensor

from regawa.data import (
    HeteroGraphBuffer,
    HeteroIndexedFactorGraph,
    heterostatedata_to_tensors,
)
from regawa.data.batch.batch_func import heterobatch
from regawa.rl.config import EXPECTED_NUM_ACTION_PARAMS

from .agent import Agent
from .types import BatchData, RolloutData


def rollout(
    agent: Agent,
    envs: VectorEnv[HeteroIndexedFactorGraph, NDArray[np.int32], NDArray[np.int32]],
    num_steps: int,
    num_envs: int,
    rng: Generator,
    device: npl.device | str,
):
    @npl.inference_mode()
    def _rollout(
        prev_obs: Mapping[str, list[HeteroIndexedFactorGraph]],
        prev_is_final: Tensor,
        b: BatchData,
        global_step: int,
    ) -> tuple[RolloutData, BatchData]:
        returns: deque[float] = deque()
        lengths: deque[int] = deque()
        obs_buf: HeteroGraphBuffer = HeteroGraphBuffer()

        is_final = prev_is_final
        obs = prev_obs
        next_obs: HeteroIndexedFactorGraph
        for step in range(0, num_steps):
            s = heterobatch(obs)
            s = heterostatedata_to_tensors(s, device)
            action, logprob, _, value = agent.sample_action_and_value(s, rng=rng)
            assert action.dim() == EXPECTED_NUM_ACTION_PARAMS
            assert action.shape[0] == num_envs
            assert logprob.dim() == 1

            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_is_final = np.logical_or(terminations, truncations)

            # add data to buffer
            b.rewards[step].copy_(npl.as_tensor(reward.reshape(-1)))
            b.actions[step] = action
            b.values[step] = value.flatten()
            b.logprobs[step] = logprob
            b.dones[step] = is_final
            obs_buf.extend(obs)
            global_step += num_envs

            obs = next_obs
            is_final.copy_(npl.as_tensor(next_is_final))

            if "episode" in infos:
                for f, r, length in zip(
                    infos["_episode"],
                    infos["episode"]["r"],
                    infos["episode"]["l"],
                    strict=False,
                ):
                    if f:
                        returns.append(r)
                        lengths.append(length)
        return RolloutData(
            obs_buf,
            obs,
            is_final,
            global_step,
            list(returns),
            list(lengths),
        ), b

    return _rollout
