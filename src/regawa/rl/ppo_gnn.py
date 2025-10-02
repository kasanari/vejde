# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import logging
import os

from regawa.policy.gnn_agent import GraphAgent
from regawa.policy.recurrent_gnn_agent import RecurrentGraphAgent

os.environ["DO_NOT_TRACK"] = "true"
import random
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TypeVar

import gymnasium as gym
import mlflow
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch

from regawa import agent_from_env
from regawa.data import HeteroObsData
from regawa.policy.load import load_agent
from .agent import Agent
from .config import Args
from .types import (
    BatchData,
    IterationCarry,
    PPOParams,
    RolloutData,
    UpdateData,
)
import tyro
import wandb
from gymnasium.spaces import Dict, MultiDiscrete
from numpy.typing import NDArray
from torch import Tensor
from tqdm import tqdm
import os

from regawa.policy import GraphAgentInterface
from regawa.data import (
    HeteroGraphBuffer,
    ObsData,
    heterostatedata,
    HeteroBatchData,
    heterostatedata_to_tensors,
)
from . import lambda_return
from .gae import gae
from .util import evaluate, save_eval_data
from .symexp import symlog

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


def minibatch_step(
    update_func: Callable[[HeteroBatchData, BatchData], UpdateData],
    device: str | npl.device,
):
    def _minibatch_step(
        start: int,
        end: int,
        b_inds: NDArray[np.int32],
        obs: HeteroGraphBuffer,
        b: BatchData,
    ):
        mb_inds = b_inds[start:end]
        minibatch = heterostatedata_to_tensors(obs.minibatch(mb_inds), device)
        return update_func(
            minibatch,
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

    return _minibatch_step


def update_step(
    batch_size: int,
    minibatch_size: int,
    mb_step: Callable[
        [int, int, NDArray[np.int32], HeteroGraphBuffer, BatchData], UpdateData
    ],
):
    def _update_step(
        obs: HeteroGraphBuffer,
        b: BatchData,
        b_inds: NDArray[np.int32],
    ) -> tuple[list[UpdateData], bool]:
        np.random.shuffle(b_inds)
        u_datas: list[UpdateData] = []
        stop_training = False
        for start in range(0, batch_size, minibatch_size):
            u_data = mb_step(
                start,
                start + minibatch_size,
                b_inds,
                obs,
                b,
            )
            u_datas.append(u_data)
            if u_data.stop_training:
                stop_training = True
                break

        return u_datas, stop_training

    return _update_step


def iteration_step(
    anneal_lr: bool,
    agent: Agent,
    batch_size: int,
    update_epochs: int,
    envs: gym.vector.SyncVectorEnv,
    optimizer: npl.optim.Optimizer,
    learning_rate: float,
    num_iterations: int,
    rollout_func: Callable[
        [dict[str, list[ObsData[V]]], Tensor, BatchData, int],
        tuple[RolloutData, BatchData],
    ],
    gae_func: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor]],
    update_func: Callable[
        [HeteroGraphBuffer, BatchData, NDArray[np.int32]],
        tuple[list[UpdateData], bool],
    ],
    lambda_return_func: Callable[[Tensor, Tensor, Tensor], Tensor],
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

        decay = 0.99
        lambda_r = lambda_return_func(b.rewards, b.values, b.dones)
        s, low_ema, high_ema = lambda_return.return_scale(
            lambda_r, carry.low_ema, carry.high_ema, decay
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
            b.actions.reshape((-1,) + envs.single_action_space.shape),  # type: ignore
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

        for epoch in range(update_epochs):
            u_data, stop_training = update_func(r_data.obs, flattened_b, b_inds)
            u_datas.extend(u_data)
            if stop_training:
                logger.info(
                    f"Early stopping at step {epoch} due to reaching max kl: {u_datas[-1].approx_kl:.2f}"
                )

        carry = IterationCarry(
            b, r_data.last_obs, r_data.last_done, r_data.global_step, low_ema, high_ema
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
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def rollout(
    agent: Agent,
    envs: gym.vector.SyncVectorEnv,
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
        for step in range(0, num_steps):
            s = heterostatedata(obs)
            s = heterostatedata_to_tensors(s, device)
            action, logprob, _, value = agent.sample_action_and_value(s)
            assert action.dim() == 2
            assert action.shape[0] == num_envs
            assert logprob.dim() == 1

            next_obs, reward, terminations, truncations, infos = envs.step(  # type: ignore
                action.cpu().numpy()  # type: ignore
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
                    infos["_episode"], infos["episode"]["r"], infos["episode"]["l"]
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


@npl.no_grad()
def approximate_kl(logprob_new: Tensor, logprob_old: Tensor) -> tuple[Tensor, Tensor]:
    # calculate approx_kl http://joschu.net/blog/kl-approx.html
    log_ratio = logprob_new - logprob_old
    ratio = npl.exp(log_ratio)
    old_approx_kl = npl.mean(-log_ratio)
    approx_kl = npl.mean((ratio - 1) - log_ratio)
    return old_approx_kl, approx_kl


def update(agent: Agent, optimizer: optim.Optimizer, params: PPOParams):
    def _update(
        s: HeteroBatchData,
        b: BatchData,
    ) -> UpdateData:
        actions, logprob_old, advantages, returns, values_old, _, _ = b
        (
            clip_coef,
            norm_adv,
            clip_range_vf,
            ent_coef,
            vf_coef,
            max_grad_norm,
            target_kl,
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

        optimizer.zero_grad()
        loss.backward()  # type: ignore
        grad_norm = nn.utils.clip_grad_norm_(
            agent.parameters(), max_grad_norm, error_if_nonfinite=True
        )

        stop_training = target_kl is not None and bool(
            (approx_kl > 1.5 * target_kl).item()
        )

        # if v_loss.item() > 500.0:
        #     per_param_grad = {
        #         k: v.grad for k, v in dict(agent.named_parameters()).items()
        #     }
        #     logger.warning(f"v_loss: {v_loss.item()}")
        #     logger.warning(f"per_param_grad: {per_param_grad}")

        optimizer.step()

        return UpdateData(
            loss,
            pg_loss,
            value_loss,
            entropy_loss,
            old_approx_kl,
            approx_kl,
            grad_norm,
            clipfrac,
            stop_training,
        )

    return _update


def main(
    envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
    run_name: str,
    args: Args,
    device: str | npl.device,
    graph_agent: GraphAgentInterface,
):
    batch_size = int(args.num_envs * args.num_steps)
    minibatch_size = int(batch_size // args.num_minibatches)
    num_iterations = args.total_timesteps // batch_size
    pbar = tqdm(total=num_iterations)
    checkpoint_period = args.checkpoint_period // batch_size
    start_time = time.time()
    artifact_name = None

    mlflow.log_params(
        {
            "batch_size": batch_size,
            "minibatch_size": minibatch_size,
            "num_iterations": num_iterations,
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
        num_iterations,
        rollout(agent, envs, args.num_steps, args.num_envs, device),
        gae(args.num_steps, args.gamma, args.gae_lambda, device),
        update_step(
            batch_size,
            minibatch_size,
            minibatch_step(
                update(
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
                device,
            ),
        ),
        lambda_return.lambda_returns(args.gamma, args.gae_lambda),
        device,
    )

    actions = npl.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape  # type: ignore
    ).to(device)
    b = BatchData(
        actions,
        npl.zeros((args.num_steps, args.num_envs)).to(device),
        npl.zeros((args.num_steps, args.num_envs)).to(device),
        npl.zeros((args.num_steps, args.num_envs)).to(device),
        npl.zeros((args.num_steps, args.num_envs)).to(device),
        npl.zeros((args.num_steps, args.num_envs)).to(device),
        npl.zeros((args.num_steps, args.num_envs)).to(device),
    )

    next_obs, _ = envs.reset(seed=args.seed)  # type: ignore
    carry = IterationCarry(
        b,
        next_obs,
        npl.zeros(args.num_envs).to(device),
        0,
    )

    if args.track:
        wandb.watch(agent, log_freq=10, log="all")  # type: ignore

    for iteration in range(1, num_iterations + 1):
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

        r = np.mean(r_data.returns) if r_data.returns else None
        length = np.mean(r_data.lengths) if r_data.lengths else None

        entropy_loss = np.mean([u.entropy_loss.item() for u in u_data])
        value_loss = np.mean([u.v_loss.item() for u in u_data])
        pg_loss = np.mean([u.pg_loss.item() for u in u_data])

        disp_r = f"{r:.2f}" if r is not None else "None"
        disp_l = f"{length:.2f}" if length is not None else "None"
        desc = f"R:{disp_r} | L:{disp_l} | ENT:{entropy_loss:.2f} | V: {value_loss:.2f} | PG: {pg_loss:.2f} | EXPL_VARIANCE:{explained_var:.2f}"
        pbar.set_description(desc)
        pbar.update(1)

        mlflow_log(
            artifact_name,
            optimizer.param_groups[0]["lr"],
            u_data,
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
        )

    envs.close()
    return agent


def mlflow_log(
    artifact_name: str | None,
    learning_rate: float,
    u_data: list[UpdateData],
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
):
    mlflow.log_artifact(
        artifact_name, artifact_path="checkpoints"
    ) if artifact_name else None
    mlflow.log_metric("charts/learning_rate", learning_rate, global_step)
    mlflow.log_metric("rollout/return_scale", return_scale.item(), global_step)
    mlflow.log_metric(
        "rollout/return_scale_low", carry.low_ema.item(), global_step
    ) if carry.low_ema is not None else None
    mlflow.log_metric(
        "rollout/return_scale_high", carry.high_ema.item(), global_step
    ) if carry.high_ema is not None else None
    mlflow.log_metric("rollout/mean_reward", b.rewards.mean().item(), global_step)
    if r is not None:
        mlflow.log_metric("rollout/mean_episodic_return", r, global_step)  # type: ignore
    if length is not None:
        mlflow.log_metric("rollout/mean_episodic_length", length, global_step)  # type: ignore
    mlflow.log_metric(
        "losses/total_loss",
        np.mean([u.loss.item() for u in u_data]),  # type: ignore
        global_step,
    )
    mlflow.log_metric(
        "losses/grad_norm",
        np.mean([u.grad_norm.item() for u in u_data]),  # type: ignore
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
        np.mean([u.old_approx_kl.item() for u in u_data]),
        global_step,
    )  # type: ignore
    mlflow.log_metric(
        "losses/approx_kl",
        np.mean([u.approx_kl.item() for u in u_data]),
        global_step,
    )  # type: ignore
    mlflow.log_metric(
        "losses/clipfrac", np.mean([u.clipfrac for u in u_data]), global_step
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


def train(args: Args | None = None, batch_id: str | None = None):
    args = tyro.cli(Args) if args is None else args
    random.seed(args.seed)
    np.random.seed(args.seed)
    npl.manual_seed(args.seed)  # type: ignore
    npl.backends.cudnn.deterministic = args.torch_deterministic
    logger.info("Attempting to connect to mlflow...")
    device = npl.device(
        "cuda:0" if npl.cuda.is_available() and args.cuda else npl.device("cpu")
    )
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"
    run_name = run_name + "__debug" if args.debug else run_name
    run_folder = create_run_folder(run_name)
    logger.addHandler(logging.FileHandler(run_folder / f"{run_name}.log"))
    agent_class = args.agent_class
    envs = (
        gym.vector.AsyncVectorEnv(
            [
                make_env(
                    args.env_id,
                )
                for _ in range(args.num_envs)
            ],
            shared_memory=False,
        )
        if args.multiprocess
        else gym.vector.SyncVectorEnv(
            [
                make_env(
                    args.env_id,
                )
                for _ in range(args.num_envs)
            ]
        )
    )

    if args.resume_from:
        agent, _ = load_agent(agent_class, args.resume_from, device)
    else:
        agent = agent_from_env(agent_class, envs, args.agent_config, device)

    logged_config = vars(args) | asdict(agent.config)
    if args.track:
        wandb.init(  # type: ignore
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=logged_config,
            name=run_name,
            save_code=True,
        )

    mlflow.enable_system_metrics_logging()
    mlflow.set_tracking_uri(uri=args.mlflow_tracking_uri)

    try:
        mlflow.create_experiment(run_name)
    except mlflow.exceptions.MlflowException:
        pass

    mlflow.set_experiment(run_name)

    with mlflow.start_run():
        logger.info(f"Connected to mlflow at {args.mlflow_tracking_uri}")
        mlflow.log_param("using_edge_attr", True)
        mlflow.log_param("using_scaling", True)
        mlflow.log_params(logged_config)
        mlflow.log_artifact(__file__)
        if Path("uv.lock").exists():
            mlflow.log_artifact("uv.lock")
        if Path("pyproject.toml").exists():
            mlflow.log_artifact("pyproject.toml")
        if batch_id:
            mlflow.log_param("batch_id", batch_id)
        run_id = mlflow.active_run().info.run_id

        agent = main(envs, run_name, args, device, agent)

        agent.agent.save_agent(run_folder / f"{run_name}.pth")
        mlflow.log_artifact(str(run_folder / f"{run_name}.pth"))

        # print(f"avg_reward: {avg_mean_reward}")
        stats, data = eval(agent.agent, args.env_id, device)
        for k, v in stats.items():
            if k != "returns":
                mlflow.log_metric(f"train_eval/{k}", v)

    stats = stats | {
        "env_id": args.env_id,
        "run_name": run_name,
        "seed": args.seed,
        "run_id": run_id,
        "weights_path": str(run_folder / f"{run_name}.pth"),
    }
    save_eval_data(data, run_folder / f"{run_name}.json")
    return stats, agent.agent


def eval(agent: GraphAgentInterface, env_id: str, device: str):
    eval_env = gym.make(
        env_id,
    )

    seeds = range(10)

    data = [
        evaluate(eval_env, agent, seed, deterministic=True, device=device)
        for seed in seeds
    ]
    rewards, *_ = zip(*data)
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
