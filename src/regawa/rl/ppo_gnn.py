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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import wandb
from gymnasium.spaces import Dict, MultiDiscrete
from numpy.typing import NDArray
from torch import Tensor
from tqdm import tqdm

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


@torch.no_grad()
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
    var_y = torch.var(y_true)
    return torch.nan if var_y == 0 else float(1 - torch.var(y_true - y_pred) / var_y)


class Agent(nn.Module):
    def __init__(
        self,
        config: AgentConfig,
        **kwargs: dict[str, Any],
    ):
        super().__init__()  # type: ignore

        self.agent = GraphAgent(
            config,
        )

    def get_value(
        self,
        s: HeteroStateData,
    ):
        value = self.agent.value(s)
        return value

    def sample_action_and_value(self, s: HeteroStateData):
        action, logprob, entropy, value, *_ = self.agent.sample(
            s,
        )
        return action, logprob, entropy, value

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
    device: str | torch.device,
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
    optimizer: torch.optim.Optimizer,
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
    device: str,
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
        with torch.no_grad():
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

        explained_var = explained_variance(b_values, b_returns)

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

        carry = IterationCarry(b, r_data.last_obs, r_data.last_done, r_data.global_step)
        return r_data, u_datas, explained_var, carry

    return _iteration_step


def gae(
    num_steps: int,
    gamma: float,
    gae_lambda: float,
    device: torch.device | str,
):
    @torch.inference_mode()
    def _gae(
        rewards: Tensor,
        dones: Tensor,
        values: Tensor,
        next_value: Tensor,
        next_step_is_terminal: Tensor,  # env will autoreset on next call to step
    ) -> tuple[Tensor, Tensor]:
        advantages = torch.zeros_like(rewards).to(device)
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


@dataclass
class Args:
    env_id: str
    domain: str
    instance: str | int
    agent_config: GNNParams
    remove_false: bool = False
    debug: bool = False
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
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
    device: torch.device | str,
):
    @torch.inference_mode()
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
                torch.Tensor(next_is_final).to(device),
            )

            # add data to buffer
            b.rewards[step] = torch.tensor(reward).to(device).view(-1)
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


@torch.no_grad()
def approximate_kl(logprob_new: Tensor, logprob_old: Tensor) -> tuple[Tensor, Tensor]:
    # calculate approx_kl http://joschu.net/blog/kl-approx.html
    log_ratio = logprob_new - logprob_old
    ratio = torch.exp(log_ratio)
    old_approx_kl = torch.mean(-log_ratio)
    approx_kl = torch.mean((ratio - 1) - log_ratio)
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
            # torch.ones_like(obs.action_mask),
            # torch.ones_like(obs.node_mask),
        )
        assert not logprob_new.isinf().any()
        assert logprob_new.dim() == 1
        assert entropy.dim() == 1

        old_approx_kl, approx_kl = approximate_kl(logprob_new, logprob_old)

        if norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss
        ratio = torch.exp(logprob_new - logprob_old)
        pg_loss1 = torch.mul(advantages, ratio)
        pg_loss2 = advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = -torch.min(pg_loss1, pg_loss2).mean()
        clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

        # Value loss
        if clip_range_vf is None:
            # No clipping
            values_pred = values_new
        else:
            values_pred = values_old + torch.clamp(
                values_new - values_old, -clip_range_vf, clip_range_vf
            )

        # Value loss using the TD(gae_lambda) target
        value_loss = nn.functional.mse_loss(returns, values_pred)

        entropy_loss = entropy.mean()
        loss = pg_loss - ent_coef * entropy_loss + value_loss * vf_coef

        assert not torch.isnan(loss).any(), loss

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
    torch.manual_seed(args.seed)  # type: ignore
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup

    agent = Agent(agent_config).to(device)

    if args.track:
        wandb.watch(agent, log_freq=10, log="all")  # type: ignore
    optimizer = optim.AdamW(
        agent.parameters(),
        lr=args.learning_rate,
        eps=1e-5,
        amsgrad=True,
        weight_decay=0.1,
    )

    # ALGO Logic: Storage setup
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape  # type: ignore
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    b = BatchData(
        actions,
        logprobs,
        torch.zeros((args.num_steps, args.num_envs)).to(device),
        torch.zeros((args.num_steps, args.num_envs)).to(device),
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
        device,
    )

    carry = IterationCarry(
        b,
        batched_hetero_dict_to_hetero_obs_list(next_obs),
        torch.zeros(args.num_envs).to(device),
        0,
    )
    for iteration in range(1, args.num_iterations + 1):
        (
            r_data,
            u_data,
            explained_var,
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

    run_name = (
        f"{Path(args.domain).name}__{Path(str(args.instance)).name}__{args.exp_name}__{args.seed}"
        if not args.debug
        else "debug"
    )

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
        mlflow.log_params(logged_config)
        mlflow.log_artifact(__file__)
        agent = main(envs, run_name, args, agent_config)

        agent.agent.save_agent(run_folder / f"{run_name}.pth")
        mlflow.log_artifact(run_folder / f"{run_name}.pth")

        eval_env = gym.make(
            args.env_id,
            domain=args.domain,
            instance=args.instance,
        )

        seeds = range(10)
        device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )
        data = [
            evaluate(eval_env, agent.agent, seed, deterministic=False, device=device)
            for seed in seeds
        ]
        rewards, *_ = zip(*data)
        avg_mean_reward = np.mean([np.mean(r) for r in rewards])
        returns = [np.sum(r) for r in rewards]
        save_eval_data(data, run_folder / f"{run_name}.json")

        mlflow.log_metric("eval/mean_reward", avg_mean_reward)

        stats = {
            "mean": np.mean(returns),
            "median": np.median(returns),
            "min": np.min(returns),
            "max": np.max(returns),
            "std": np.std(returns),
        }

        for k, v in stats.items():
            mlflow.log_metric(f"eval/return_{k}", v)

    print(stats)
    print(f"avg_reward: {avg_mean_reward}")

    return stats, agent
