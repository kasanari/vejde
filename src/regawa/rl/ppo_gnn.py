# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import logging
import os
import random
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import gymnasium as gym
import mlflow
import mlflow.pytorch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import tyro
import wandb
from gymnasium.spaces import Dict, MultiDiscrete
from numpy.typing import NDArray
from torch import Tensor
from tqdm import tqdm

from regawa.rl import lambda_return
from regawa.rl.gae import gae
import regawa.wrappers.gym_utils as model_utils
from regawa import GNNParams
from regawa.gnn import AgentConfig, GraphAgent, HeteroStateData
from regawa.gnn.data import (
    HeteroGraphBuffer,
    ObsData,
    batched_hetero_dict_to_hetero_obs_list,
    heterostatedata,
)
from regawa.gnn.gnn_agent import heterostatedata_to_tensors
from regawa.rl.util import evaluate, save_eval_data

logger = logging.getLogger(__name__)


npl = torch


def symlog(x: Tensor):
    # return x
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x: Tensor):
    # return x
    x = torch.clip(
        x, -20, 20
    )  # Clipped to prevent extremely rare occurence where critic throws a huge value
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class RolloutData(NamedTuple):
    obs: HeteroGraphBuffer
    last_obs: dict[str, list[ObsData]]
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


class Agent(nn.Module):
    def __init__(
        self,
        config: AgentConfig,
        **kwargs: dict[str, Any],
    ):
        super().__init__()  # type: ignore

        self.agent = GraphAgent(
            config,
            None,
        )

    def get_value(
        self,
        s: HeteroStateData,
    ):
        value = self.agent.value(s)
        return symexp(value)

    def sample_action_and_value(self, s: HeteroStateData):
        action, logprob, entropy, value, *_ = self.agent.sample(
            s,
        )
        return action, logprob, entropy, symexp(value)

    def evaluate_action_and_value(
        self,
        action: Tensor,
        s: HeteroStateData,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # num_graphs = batch_idx.max() + 1
        # action_mask = action_mask.reshape(num_graphs, -1)
        logprob, entropy, value, *_ = self.agent.forward(
            action,
            s,
            # num_graphs,
            # action_mask,
            # node_mask,
        )
        entropy = entropy.unsqueeze(0)
        return (
            logprob,
            entropy,
            value,
        )


def minibatch_step(
    minibatch_size: int,
    update_func: Callable[[HeteroStateData, BatchData], UpdateData],
    device: str | npl.device,
):
    def _minibatch_step(
        start: int,
        b_inds: NDArray[np.int32],
        obs: HeteroGraphBuffer,
        b: BatchData,
    ):
        end = start + minibatch_size
        mb_inds = b_inds[start:end]
        minibatch = obs.minibatch(mb_inds)
        minibatch = heterostatedata_to_tensors(minibatch, device)
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
        [int, NDArray[np.int32], HeteroGraphBuffer, BatchData], UpdateData
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
        [dict[str, list[ObsData]], Tensor, BatchData, int],
        tuple[RolloutData, BatchData],
    ],
    gae_func: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor]],
    update_func: Callable[
        [HeteroGraphBuffer, BatchData, NDArray[np.int32]],
        tuple[list[UpdateData], bool],
    ],
    lambda_returns: Callable[[Tensor, Tensor, Tensor], Tensor],
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
        b_logprobs = b.logprobs.reshape(-1)
        b_actions = b.actions.reshape((-1,) + envs.single_action_space.shape)  # type: ignore
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = b.values.reshape(-1)

        decay = 0.99
        lambda_r = lambda_returns(b.rewards, b.values, b.dones)
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
            b_actions,
            b_logprobs,
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
                print(
                    f"Early stopping at step {epoch} due to reaching max kl: {u_datas[-1].approx_kl:.2f}"
                )

        carry = IterationCarry(
            b, r_data.last_obs, r_data.last_done, r_data.global_step, low_ema, high_ema
        )
        return r_data, u_datas, explained_variance(b_values, b_returns), s, carry

    return _iteration_step


@dataclass
class Args:
    env_id: str
    domain: str
    instance: str | int | list[int]
    agent_config: GNNParams
    remove_false: bool = False
    debug: bool = False
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `npl.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    total_timesteps: int = 2000
    """total timesteps of the experiments"""
    learning_rate: float = 1.0e-2
    """the learning rate of the optimizer"""
    weight_decay: float = 0.1
    """the weight decay of the optimizer"""
    num_envs: int = 5
    """the number of parallel game environments"""
    num_steps: int = 20
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 1.0
    """the discount factor gamma"""
    gae_lambda: float = 0.0
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 10
    """the number of mini-batches"""
    update_epochs: int = 8
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 1.0
    """coefficient of the value function"""
    max_grad_norm: float = 1.0
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(
    env_id: str,
    domain: str,
    instance: str | int,
    remove_false: bool,
):
    def thunk() -> gym.Env[Dict, MultiDiscrete]:
        env: gym.Env[Dict, MultiDiscrete] = gym.make(  # type: ignore
            env_id,
            domain=domain,
            instance=instance,
            remove_false=remove_false,
            # add_inverse_relations=False,
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
        prev_obs: dict[str, list[ObsData]],
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

            next_obs_dict, reward, terminations, truncations, infos = envs.step(  # type: ignore
                action.cpu().numpy()  # type: ignore
            )
            next_is_final = np.logical_or(terminations, truncations)
            next_obs, next_is_final = (
                batched_hetero_dict_to_hetero_obs_list(next_obs_dict),  # type: ignore
                npl.as_tensor(next_is_final).to(device),
            )

            # add data to buffer
            b.rewards[step] = npl.as_tensor(reward).to(device).view(-1)
            b.actions[step] = action
            b.values[step] = value.flatten()
            b.logprobs[step] = logprob
            b.dones[step] = is_final
            obs_buf.extend(obs)
            global_step += num_envs

            obs = next_obs
            is_final = next_is_final

            if "episode" in infos:
                for next_is_final, r, l in zip(
                    infos["_episode"], infos["episode"]["r"], infos["episode"]["l"]
                ):
                    if next_is_final:
                        returns.append(r)
                        lengths.append(l)
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
        s: HeteroStateData,
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


class IterationCarry(NamedTuple):
    b: BatchData
    next_obs: dict[str, list[ObsData]]
    next_done: Tensor
    global_step: int
    low_ema: Tensor | None = None
    high_ema: Tensor | None = None


def main(
    envs: gym.vector.SyncVectorEnv,
    run_name: str,
    args: Args,
    agent_config: AgentConfig,
):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    npl.manual_seed(args.seed)  # type: ignore
    npl.backends.cudnn.deterministic = args.torch_deterministic

    device = npl.device("cuda" if npl.cuda.is_available() and args.cuda else "cpu")

    # env setup

    agent = Agent(agent_config).to(device)

    if args.track:
        wandb.watch(agent, log_freq=10, log="all")  # type: ignore
    optimizer = optim.AdamW(
        agent.parameters(),
        lr=args.learning_rate,
        eps=1e-5,
        amsgrad=True,
        weight_decay=args.weight_decay,
    )

    # ALGO Logic: Storage setup
    actions = npl.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape  # type: ignore
    ).to(device)
    logprobs = npl.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = npl.zeros((args.num_steps, args.num_envs)).to(device)
    dones = npl.zeros((args.num_steps, args.num_envs)).to(device)
    values = npl.zeros((args.num_steps, args.num_envs)).to(device)

    b = BatchData(
        actions,
        logprobs,
        npl.zeros((args.num_steps, args.num_envs)).to(device),
        npl.zeros((args.num_steps, args.num_envs)).to(device),
        values,
        rewards,
        dones,
    )

    # TRY NOT TO MODIFY: start the game

    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)  # type: ignore

    ppo_params = PPOParams(
        args.clip_coef,
        args.norm_adv,
        args.clip_vloss,
        args.ent_coef,
        args.vf_coef,
        args.max_grad_norm,
        args.target_kl,
    )

    pbar = tqdm(total=args.num_iterations)

    update_func = update(agent, optimizer, ppo_params)
    mb_func = minibatch_step(args.minibatch_size, update_func, device)
    update_step_func = update_step(args.batch_size, args.minibatch_size, mb_func)
    gae_func = gae(args.num_steps, args.gamma, args.gae_lambda, device)
    lambda_returns_func = lambda_return.lambda_returns(args.gamma, args.gae_lambda)

    iter_step_func = iteration_step(
        args.anneal_lr,
        agent,
        args.batch_size,
        args.update_epochs,
        envs,
        optimizer,
        args.learning_rate,
        args.num_iterations,
        rollout(agent, envs, args.num_steps, args.num_envs, device),
        gae_func,
        update_step_func,
        lambda_returns_func,
        device,
    )

    carry = IterationCarry(
        b,
        batched_hetero_dict_to_hetero_obs_list(next_obs),
        npl.zeros(args.num_envs).to(device),
        0,
    )
    for iteration in range(1, args.num_iterations + 1):
        (
            r_data,
            u_data,
            explained_var,
            return_scale,
            carry,
        ) = iter_step_func(iteration, carry)

        r = np.mean(r_data.returns) if r_data.returns else None
        length = np.mean(r_data.lengths) if r_data.lengths else None
        global_step = carry.global_step
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        mlflow.log_metric(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )

        entropy_loss = np.mean([u.entropy_loss.item() for u in u_data])
        value_loss = np.mean([u.v_loss.item() for u in u_data])
        pg_loss = np.mean([u.pg_loss.item() for u in u_data])

        disp_r = f"{r:.2f}" if r is not None else "None"
        disp_l = f"{length:.2f}" if length is not None else "None"
        desc = f"R:{disp_r} | L:{disp_l} | ENT:{entropy_loss:.2f} | V: {value_loss:.2f} | PG: {pg_loss:.2f} | EXPL_VARIANCE:{explained_var:.2f}"
        pbar.set_description(desc)
        pbar.update(1)

        mlflow.log_metric("rollout/return_scale", return_scale.item(), global_step)
        mlflow.log_metric("rollout/return_scale_low", carry.low_ema.item(), global_step)
        mlflow.log_metric(
            "rollout/return_scale_high", carry.high_ema.item(), global_step
        )
        mlflow.log_metric("rollout/mean_reward", b.rewards.mean().item(), global_step)
        if r is not None:
            mlflow.log_metric("rollout/mean_episodic_return", r, global_step)  # type: ignore
        if length is not None:
            mlflow.log_metric("rollout/mean_episodic_length", length, global_step)  # type: ignore
        mlflow.log_metric(
            "losses/total_loss", np.mean([u.loss.item() for u in u_data]), global_step
        )  # type: ignore
        mlflow.log_metric(
            "losses/grad_norm",
            np.mean([u.grad_norm.item() for u in u_data]),
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

    envs.close()
    return agent


def setup(args: Args | None = None):
    args = tyro.cli(Args) if args is None else args

    print("Attempting to connect to mlflow...")
    tracking_uri = "http://127.0.0.1:5000" if not args.debug else ""
    mlflow.set_tracking_uri(uri=tracking_uri)
    print(f"Connected to mlflow at {tracking_uri}")

    run_name = f"{Path(args.domain).name}__{Path(str(args.instance)).name}__{args.exp_name}__{args.seed}"
    run_name = run_name + "__debug" if args.debug else run_name

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.domain,
                args.instance,
                args.remove_false,
            )
            for _ in range(args.num_envs)
        ],
    )

    n_types = model_utils.n_types(envs.single_observation_space)  # type: ignore
    n_relations = model_utils.n_relations(envs.single_observation_space)  # type: ignore
    n_actions = model_utils.n_actions(envs.single_action_space)  # type: ignore

    agent_config = AgentConfig(
        n_types,
        n_relations,
        n_actions,
        args.remove_false,
        args.agent_config,
        model_utils.max_arity(envs.single_observation_space),  # type: ignore
    )

    logged_config = vars(args) | asdict(agent_config)
    if args.track:
        wandb.init(  # type: ignore
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=logged_config,
            name=run_name,
            save_code=True,
        )

    try:
        mlflow.create_experiment(run_name)
    except Exception:
        pass

    mlflow.set_experiment(run_name)

    runs_folder = Path("runs")
    runs_folder.mkdir(exist_ok=True)
    run_folder = runs_folder / run_name
    run_folder.mkdir(exist_ok=True)

    logger.addHandler(logging.FileHandler(run_folder / f"{run_name}.log"))

    with mlflow.start_run():
        mlflow.log_param("using_edge_attr", True)
        mlflow.log_param("using_scaling", True)
        mlflow.log_params(logged_config)
        mlflow.log_artifact(__file__)
        if Path("uv.lock").exists():
            mlflow.log_artifact("uv.lock")
        if Path("pyproject.toml").exists():
            mlflow.log_artifact("pyproject.toml")
        agent = main(envs, run_name, args, agent_config)

        agent.agent.save_agent(run_folder / f"{run_name}.pth")
        mlflow.log_artifact(run_folder / f"{run_name}.pth")

        eval_env = gym.make(
            args.env_id,
            domain=args.domain,
            instance=args.instance,
            remove_false=args.remove_false,
        )

        seeds = range(10)
        device = npl.device("cuda" if npl.cuda.is_available() and args.cuda else "cpu")
        data = [
            evaluate(eval_env, agent.agent, seed, deterministic=True, device=device)
            for seed in seeds
        ]
        rewards, *_ = zip(*data)
        avg_mean_reward = np.mean([np.mean(r) for r in rewards])
        returns = [np.sum(r) for r in rewards]
        save_eval_data(data, run_folder / f"{run_name}.json")

        mlflow.log_metric("eval/mean_reward", avg_mean_reward)

        stats = {
            "mean": np.mean(returns).item(),
            "median": np.median(returns).item(),
            "min": np.min(returns).item(),
            "max": np.max(returns).item(),
            "std": np.std(returns).item(),
        }

        for k, v in stats.items():
            mlflow.log_metric(f"eval/return_{k}", v)

    print(stats)
    print(f"avg_reward: {avg_mean_reward}")

    return stats, agent
