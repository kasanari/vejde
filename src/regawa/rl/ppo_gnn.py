# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import logging
import os
from functools import partial

from regawa.data import (
    HeteroGraphBuffer,
    HeteroObsData,
    TorchHeteroBatchData,
    heterostatedata,
    heterostatedata_to_tensors,
)

os.environ["DO_NOT_TRACK"] = "true"
import random
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TypeVar

import gymnasium as gym
import mlflow  # type: ignore
import numpy as np
import torch
import tyro
from gymnasium.spaces import Dict, MultiDiscrete
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from numpy.typing import NDArray
from torch import Tensor, nn, optim
from tqdm import tqdm

from regawa import (
    GraphAgent,
    GraphAgentInterface,
    RecurrentGraphAgent,
    agent_from_env,
    load_agent,
)

from . import lambda_return
from .agent import Agent
from .config import Args, ConcurrencySetting
from .gae import gae
from .symexp import symlog
from .types import (
    BatchData,
    IterationCarry,
    LossData,
    PPOParams,
    RolloutData,
    UpdateData,
)
from .util import evaluate, writable_eval_data

logger = logging.getLogger(__name__)

V = TypeVar("V", np.float32, np.bool_, np.int64)
npl = torch


@npl.no_grad()
def explained_variance(y_pred: Tensor, y_true: Tensor) -> float:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = npl.var(y_true)
    return npl.nan if var_y == 0 else float(1 - npl.var(y_true - y_pred) / var_y)


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


def iteration_step(
    anneal_lr: bool,
    agent: Agent,
    batch_size: int,
    update_epochs: int,
    envs: gym.vector.VectorEnv[HeteroObsData, NDArray[np.int32], NDArray[np.int32]],
    optimizer: npl.optim.Optimizer,
    learning_rate: float,
    num_iterations: int,
    rollout_func: Callable[
        [dict[str, list[HeteroObsData]], Tensor, BatchData, int],
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
                heterostatedata(r_data.last_obs), device
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

        b_returns = symlog(b_returns)
        b_values = symlog(b_values)

        # plt.hist(b_returns.cpu().numpy(), bins=100)
        # plt.savefig("returns_symlog.png")
        # plt.close()

        flattened_b = BatchData(
            b.actions.reshape((-1, *envs.single_action_space.shape)),  # type: ignore
            b.logprobs.reshape(-1),
            b_advantages,
            b_returns,
            b_values,
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
            b,
            r_data.last_obs,
            r_data.last_done,
            r_data.global_step,
            total_num_updates,
            low_ema,
            high_ema,
        )
        return r_data, u_datas, explained_variance(b_values, b_returns), s, carry

    return _iteration_step


def make_env(
    env_id: str,
):
    def thunk() -> gym.Env[Dict, MultiDiscrete]:
        env: gym.Env[Dict, MultiDiscrete] = gym.make(  # type: ignore
            env_id,
        )
        return gym.wrappers.RecordEpisodeStatistics(env)

    return thunk


EXPECTED_NUM_ACTION_PARAMS = 2


def rollout(
    agent: Agent,
    envs: gym.vector.VectorEnv[HeteroObsData, NDArray[np.int32], NDArray[np.int32]],
    num_steps: int,
    num_envs: int,
    device: npl.device | str,
):
    @npl.inference_mode()
    def _rollout(
        prev_obs: Mapping[str, list[HeteroObsData]],
        prev_is_final: Tensor,
        b: BatchData,
        global_step: int,
    ) -> tuple[RolloutData, BatchData]:
        returns: deque[float] = deque()
        lengths: deque[int] = deque()
        obs_buf: HeteroGraphBuffer = HeteroGraphBuffer()

        is_final = prev_is_final
        obs = prev_obs
        next_obs: HeteroObsData
        for step in range(0, num_steps):
            s = heterostatedata(obs)
            s = heterostatedata_to_tensors(s, device)
            action, logprob, _, value = agent.sample_action_and_value(s)
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


@npl.no_grad()  # type: ignore
def approximate_kl(logprob_new: Tensor, logprob_old: Tensor) -> tuple[Tensor, Tensor]:
    # calculate approx_kl http://joschu.net/blog/kl-approx.html
    log_ratio = logprob_new - logprob_old
    ratio = npl.exp(log_ratio)
    old_approx_kl = npl.mean(-log_ratio)
    approx_kl = npl.mean((ratio - 1) - log_ratio)
    return old_approx_kl, approx_kl


def calculate_loss(agent: Agent, params: PPOParams):
    def f(s: TorchHeteroBatchData, b: BatchData):
        actions, logprob_old, advantages, returns, values_old, _, _ = b
        (
            clip_coef,
            norm_adv,
            clip_range_vf,
            ent_coef,
            vf_coef,
            _,
            _,
        ) = params

        logprob_new, entropy, values_new = agent.evaluate_action_and_value(
            actions,
            s,
            # npl.ones_like(obs.action_mask),
            # npl.ones_like(obs.node_mask),
        )
        assert not logprob_new.isinf().any()
        assert logprob_new.dim() == 1
        assert entropy.dim() == 1

        old_approx_kl, approx_kl = approximate_kl(logprob_new, logprob_old)

        if norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss
        ratio = npl.exp(logprob_new - logprob_old)
        pg_loss1 = advantages * ratio
        pg_loss2 = advantages * npl.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = -npl.min(pg_loss1, pg_loss2).mean()
        clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

        # Value loss
        if clip_range_vf is None:
            # No clipping
            values_pred = values_new
        else:
            values_pred = values_old + npl.clamp(
                values_new - values_old, -clip_range_vf, clip_range_vf
            )

        # Value loss using the TD(gae_lambda) target
        value_loss = nn.functional.mse_loss(returns, values_pred)

        entropy_loss = entropy.mean()
        loss = pg_loss - ent_coef * entropy_loss + value_loss * vf_coef

        assert not npl.isnan(loss).any(), loss
        return LossData(
            loss,
            pg_loss,
            value_loss,
            entropy_loss,
            old_approx_kl,
            approx_kl,
            clipfrac,
        )

    return f


MAX_ALLOWED_GRAD_NORM = 100.0


def update(agent: Agent, optimizer: optim.Optimizer, params: PPOParams):
    loss_func = calculate_loss(agent, params)

    def _update(
        s: TorchHeteroBatchData,
        b: BatchData,
    ) -> UpdateData:
        loss = loss_func(s, b)

        assert not npl.isnan(loss.loss).any(), loss

        optimizer.zero_grad()
        loss.loss.backward()  # type: ignore

        # per_param_grad = {
        #         k: v.grad for k, v in dict(agent.named_parameters()).items()
        #     }
        # per_param_grad_norm = {k: v.norm().item() if v is not None else None for k, v in per_param_grad.items()}
        # sorted_per_param_grad = sorted(
        #         per_param_grad_norm.items(), key=lambda item: item[1] if item[1] is not None else -1, reverse=True
        # )

        grad_norm = nn.utils.clip_grad_norm_(
            agent.parameters(), params.max_grad_norm, error_if_nonfinite=True
        )

        stop_training = params.target_kl is not None and bool(
            (loss.approx_kl > 1.5 * params.target_kl).item()
        )

        if grad_norm.item() > MAX_ALLOWED_GRAD_NORM:
            logger.warning(f"grad_norm: {grad_norm.item()}")
            # logger.warning(f"per_param_grad: {per_param_grad}")

        optimizer.step()

        return UpdateData(
            loss,
            grad_norm,
            stop_training,
        )

    return _update


def main(
    envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
    run_name: str,
    args: Args,
    device: str | npl.device,
    graph_agent: GraphAgentInterface,
    rng: np.random.Generator,
):
    batch_size = int(args.num_envs * args.rollout_length)
    minibatch_size = int(batch_size // args.num_minibatches)
    num_rollouts = 0

    if args.total_timesteps:
        num_rollouts = args.total_timesteps // batch_size

    if args.total_updates:
        inner_updates = (batch_size // minibatch_size) * args.update_epochs
        assert (
            args.total_updates % inner_updates == 0
        ), "total_updates must be multiple of (batch_size / minibatch_size) * update_epochs"
        num_rollouts = args.total_updates // inner_updates

    pbar = tqdm(total=num_rollouts)
    checkpoint_period = args.checkpoint_period // batch_size
    start_time = time.time()
    artifact_name = None

    mlflow.log_params(  # type: ignore
        {
            "batch_size": batch_size,
            "minibatch_size": minibatch_size,
            "num_iterations": num_rollouts,
        }
    )

    agent = Agent(graph_agent)
    agent = agent.to(device)
    optimizer = optim.AdamW(
        agent.parameters(),
        lr=args.learning_rate,
        eps=1e-5,
        amsgrad=True,
        weight_decay=args.weight_decay,
    )
    iter_step_func = iteration_step(
        args.anneal_lr,
        agent,
        batch_size,
        args.update_epochs,
        envs,
        optimizer,
        args.learning_rate,
        num_rollouts,
        rollout(agent, envs, args.rollout_length, args.num_envs, device),
        gae(args.rollout_length, args.gamma, args.gae_lambda, device),
        update_step(
            batch_size,
            minibatch_size,
            device=device,
            update_func=update(
                agent,
                optimizer,
                PPOParams(
                    args.clip_coef,
                    args.norm_adv,
                    args.clip_vloss,
                    args.ent_coef,
                    args.vf_coef,
                    args.max_grad_norm,
                    args.target_kl,
                ),
            ),
            rng=rng,
        ),
        lambda_return.lambda_returns(args.gamma, args.gae_lambda),
        device=device,
        ema_decay=args.ema_decay,
    )

    actions = npl.zeros(
        (args.rollout_length, args.num_envs, *envs.single_action_space.shape)  # type: ignore
    ).to(device)
    b = BatchData(
        actions,
        npl.zeros((args.rollout_length, args.num_envs)).to(device),
        npl.zeros((args.rollout_length, args.num_envs)).to(device),
        npl.zeros((args.rollout_length, args.num_envs)).to(device),
        npl.zeros((args.rollout_length, args.num_envs)).to(device),
        npl.zeros((args.rollout_length, args.num_envs)).to(device),
        npl.zeros((args.rollout_length, args.num_envs)).to(device),
    )

    next_obs, _ = envs.reset(seed=args.seed)  # type: ignore
    carry = IterationCarry(
        b,
        next_obs,
        npl.zeros(args.num_envs).to(device),
        0,
        0,
    )

    for iteration in range(1, num_rollouts + 1):
        (
            r_data,
            u_data,
            explained_var,
            return_scale,
            carry,
        ) = iter_step_func(iteration, carry)

        if checkpoint_period > 0 and iteration % checkpoint_period == 0:
            artifact_name = f"runs/{run_name}/checkpoint_{iteration*batch_size}.pth"
            agent.agent.save_agent(artifact_name)
            # hard link to "checkpoint_latest.pth"
            latest_path = f"runs/{run_name}/checkpoint_latest.pth"
            if os.path.exists(latest_path):
                os.remove(latest_path)
            os.link(artifact_name, latest_path)

        r = float(np.mean(r_data.returns)) if r_data.returns else None
        length = float(np.mean(r_data.lengths)) if r_data.lengths else None

        loss_data = [u.loss for u in u_data]
        grad_norm = float(np.mean([u.grad_norm.item() for u in u_data]))
        total_loss = float(np.mean([u.loss.item() for u in loss_data]))
        entropy_loss = float(np.mean([u.entropy_loss.item() for u in loss_data]))
        value_loss = float(np.mean([u.v_loss.item() for u in loss_data]))
        pg_loss = float(np.mean([u.pg_loss.item() for u in loss_data]))

        disp_r = f"{r:.2f}" if r is not None else "N/A"
        disp_l = f"{length:.2f}" if length is not None else "N/A"
        desc = f"R:{disp_r} | L:{disp_l} | ENT:{entropy_loss:.2f} | V: {value_loss:.2f} | PG: {pg_loss:.2f} | EXPL_VARIANCE:{explained_var:.2f}"
        pbar.set_description(desc)
        pbar.update(1)

        if mlflow.active_run() is not None:  # type: ignore
            mlflow_log(
                artifact_name,
                optimizer.param_groups[0]["lr"],
                u_data,
                total_loss,
                grad_norm,
                value_loss,
                pg_loss,
                entropy_loss,
                explained_var,
                return_scale,
                carry,
                b,
                r,
                length,
                carry.global_step,
                start_time,
                carry.num_updates,
            )

    envs.close()
    return agent


def mlflow_log(
    artifact_name: str | None,
    learning_rate: float,
    u_data: list[UpdateData],
    total_loss: float,
    grad_norm: float,
    value_loss: float,
    pg_loss: float,
    entropy_loss: float,
    explained_var: float,
    return_scale: Tensor,
    carry: IterationCarry,
    b: BatchData,
    r: float | None,
    length: float | None,
    global_step: int,
    start_time: float,
    gradient_steps: int,
):
    mlflow.log_artifact(
        artifact_name, artifact_path="checkpoints"
    ) if artifact_name else None
    mlflow.log_metric("charts/learning_rate", learning_rate, global_step)
    mlflow.log_metric("rollout/return_scale", return_scale.item(), global_step)
    mlflow.log_metric(
        "rollout/return_scale_low", carry.low_ema.item(), global_step
    ) if carry.low_ema is not None else None
    mlflow.log_metric("charts/num_gradient_steps", gradient_steps, global_step)
    mlflow.log_metric(
        "rollout/return_scale_high", carry.high_ema.item(), global_step
    ) if carry.high_ema is not None else None
    mlflow.log_metric("rollout/mean_reward", b.rewards.mean().item(), global_step)

    mlflow.log_metric("rollout/num_resets", b.dones.sum().item(), global_step)

    if r is not None:
        mlflow.log_metric("rollout/mean_episodic_return", r, global_step)  # type: ignore
    if length is not None:
        mlflow.log_metric("rollout/mean_episodic_length", length, global_step)  # type: ignore
    mlflow.log_metric(
        "losses/total_loss",
        total_loss,  # type: ignore
        global_step,
    )
    mlflow.log_metric(
        "losses/grad_norm",
        grad_norm,  # type: ignore
        global_step,
    )  # type: ignore
    mlflow.log_metric("losses/value_loss", value_loss, global_step)  # type: ignore
    mlflow.log_metric(
        "losses/policy_loss",
        pg_loss,
        global_step,
    )  # type: ignore
    mlflow.log_metric(
        "losses/entropy",
        entropy_loss,
        global_step,
    )  # type: ignore
    mlflow.log_metric(
        "losses/old_approx_kl",
        np.mean([u.loss.old_approx_kl.item() for u in u_data]),
        global_step,
    )  # type: ignore
    mlflow.log_metric(
        "losses/approx_kl",
        np.mean([u.loss.approx_kl.item() for u in u_data]),
        global_step,
    )  # type: ignore
    mlflow.log_metric(
        "losses/clipfrac", np.mean([u.loss.clipfrac for u in u_data]), global_step
    )  # type: ignore
    mlflow.log_metric("losses/explained_variance", explained_var, global_step)  # type: ignore
    mlflow.log_metric(
        "charts/SPS", int(global_step / (time.time() - start_time)), global_step
    )


def create_run_folder(run_name: str) -> Path:
    runs_folder = Path("runs")
    runs_folder.mkdir(exist_ok=True)
    run_folder = runs_folder / run_name
    run_folder.mkdir(exist_ok=True)
    return run_folder


AGENT_CLASSES: dict[str, type[GraphAgentInterface]] = {
    c.__name__: c
    for c in [
        GraphAgent,
        RecurrentGraphAgent,
    ]
}


env_settings = {
    ConcurrencySetting.MULTI: partial(
        AsyncVectorEnv,
        shared_memory=False,
    ),
    ConcurrencySetting.SINGLE: SyncVectorEnv,
}


def train(args: Args | None = None):
    args = tyro.cli(Args) if args is None else args
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    npl.manual_seed(args.seed)  # type: ignore
    npl.backends.cudnn.deterministic = args.torch_deterministic
    torch.use_deterministic_algorithms(args.torch_deterministic)
    run_name = f"{args.env_id}__ppo"
    run_folder = create_run_folder(run_name)
    logger.addHandler(logging.FileHandler(run_folder / f"{run_name}.log", mode="w"))
    logger.info("Attempting to connect to mlflow...")
    device = npl.device(
        "cuda:0" if npl.cuda.is_available() and args.cuda else npl.device("cpu")
    )
    logger.info(f"Using device: {device}")
    agent_class = AGENT_CLASSES[args.agent_class]

    envs = env_settings[args.multiprocess](
        [make_env(args.env_id) for _ in range(args.num_envs)]
    )

    if args.resume_from:
        agent, *_ = load_agent(agent_class, args.resume_from, device)
    else:
        agent = agent_from_env(agent_class, envs, args.agent_config, device)

    logged_config = vars(args) | asdict(agent.config)

    try:
        agent = main(envs, run_name, args, device, agent, rng=rng)
    except Exception as e:
        logger.exception("Exception during training:")
        raise e

    # print(f"avg_reward: {avg_mean_reward}")
    stats, data = evaluate_agent(agent.agent, args.env_id, device)

    stats = stats | {
        "env_id": args.env_id,
        "config": logged_config,
        "run_name": run_name,
        "seed": args.seed,
        "run_folder": str(run_folder),
        "eval_data": writable_eval_data(data),
    }

    return stats, agent.agent


def evaluate_agent(agent: GraphAgentInterface, env_id: str, device: str):
    eval_env = gym.make(  # type: ignore
        env_id,
    )

    seeds = range(10)

    data = [
        evaluate(eval_env, agent, seed, deterministic=True, device=device)
        for seed in seeds
    ]
    rewards, *_ = zip(*data, strict=False)
    avg_mean_reward = np.mean([np.mean(r) for r in rewards])
    returns = [np.sum(r).item() for r in rewards]

    stats = {
        "return_mean": np.mean(returns).item(),
        "return_median": np.median(returns).item(),
        "return_min": np.min(returns).item(),
        "return_max": np.max(returns).item(),
        "return_std": np.std(returns).item(),
        "mean_reward": avg_mean_reward.item(),
    }

    stats["returns"] = returns
    return stats, data
