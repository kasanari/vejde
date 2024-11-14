# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
from collections import deque
import os
from pathlib import Path
import random
import time
from dataclasses import dataclass

import gymnasium as gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical

from torch import Tensor
import wandb
from tqdm import tqdm
from typing import Any

from torch_geometric.data import Data, Batch

from kg_gnn import KGGNNAgent
from wrappers import kg_wrapper
import mlflow
import mlflow.pytorch


def dict_to_data(obs: dict[str, tuple[Any]], num_envs: int) -> list[Data]:
    return [
        Data(
            x=torch.as_tensor(obs["nodes"][n], dtype=torch.int64),
            edge_index=torch.as_tensor(obs["edge_index"][n], dtype=torch.int64).T,
            edge_attr=torch.as_tensor(obs["edge_attr"][n], dtype=torch.int64),
        )
        for n in range(num_envs)
    ]


def save_agent(
    envs: gym.vector.SyncVectorEnv, agent: nn.Module, config: dict[str, Any], path: str
):
    state_dict = agent.state_dict()
    to_save = {}
    n_types = envs.single_observation_space.spaces["nodes"].feature_space.n
    n_relations = envs.single_observation_space.spaces["edge_attr"].feature_space.n
    n_actions = envs.single_action_space.nvec[1]
    to_save["config"] = {
        "n_types": n_types,
        "n_relations": n_relations,
        "n_actions": n_actions,
        **config,
    }
    to_save["state_dict"] = state_dict
    torch.save(to_save, path)


def create_batch(data: list[Data]):
    return Batch.from_data_list(data)


@torch.inference_mode()
def gae(
    rewards: Tensor,
    dones: Tensor,
    values: Tensor,
    next_value: Tensor,
    next_done: Tensor,
    num_steps: int,
    gamma: float,
    gae_lambda: float,
    device: torch.device | str,
) -> tuple[Tensor, Tensor]:
    advantages = torch.zeros_like(rewards).to(device)
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = (
            delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        )
    returns = advantages + values
    return advantages, returns


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "SysAdmin_MDP_ippc2011"  # "Elevators_MDP_ippc2011"
    """the id of the environment"""
    total_timesteps: int = 10000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-3
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 80
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.1
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 1
    """the number of mini-batches"""
    update_epochs: int = 2
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.1
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
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


def make_env(domain: str, idx: int, capture_video: bool, run_name: str):
    instance = 1

    def thunk():
        env = gym.make(
            "KGRDDLGraphWrapper-v0",
            domain=domain,
            instance=instance,
            add_inverse_relations=False,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(
        self, envs: gym.vector.SyncVectorEnv, device: str | None = None, **kwargs
    ):
        super().__init__()
        n_types = envs.single_observation_space.spaces["nodes"].feature_space.n
        n_relations = envs.single_observation_space.spaces["edge_attr"].feature_space.n
        n_actions = envs.single_action_space.nvec[1]
        self.gnn_agent = KGGNNAgent(n_types, n_relations, n_actions, **kwargs)

    def get_value(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch_idx: Tensor,
    ):
        value = self.gnn_agent.value(x, edge_index, edge_attr, batch_idx)
        return value

    def sample_action_and_value(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch_idx: Tensor
    ):
        action, logprob, entropy, value = self.gnn_agent.sample_action(
            x,
            edge_index,
            edge_attr,
            batch_idx,
        )
        return action, logprob, entropy, value

    def evaluate_action_and_value(
        self,
        action: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch_idx: Tensor,
        # action_mask: Tensor,
        # node_mask: Tensor,
    ):
        # num_graphs = batch_idx.max() + 1
        # action_mask = action_mask.reshape(num_graphs, -1)
        logprob, entropy, value = self.gnn_agent.forward(
            action,
            x,
            edge_index,
            edge_attr,
            batch_idx,
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


@torch.inference_mode()
def rollout(
    agent: Agent,
    envs: gym.vector.SyncVectorEnv,
    next_obs: list[Data],
    next_done: Tensor,
    obs: deque[Data],
    dones: Tensor,
    rewards: Tensor,
    actions: Tensor,
    logprobs: Tensor,
    values: Tensor,
    num_steps: int,
    num_envs: int,
    device: torch.device | str,
    global_step: int,
) -> tuple[int, list[float], list[int]]:
    returns: deque[float] = deque()
    lengths: deque[int] = deque()
    for step in range(0, num_steps):
        global_step += num_envs
        obs.extend(next_obs)
        dones[step] = next_done

        # ALGO LOGIC: action logic
        batch = create_batch(next_obs)
        action, logprob, _, value = agent.sample_action_and_value(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            # batch.action_mask,
            # batch.node_mask,
        )
        assert action.dim() == 2
        assert action.shape[0] == num_envs
        assert logprob.dim() == 1
        assert value.dim() == 2
        values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, terminations, truncations, infos = envs.step(
            action.cpu().numpy()
        )
        next_done = np.logical_or(terminations, truncations)
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = (
            dict_to_data(next_obs, num_envs),
            torch.Tensor(next_done).to(device),
        )

        if "episode" in infos:
            for is_final, r, l in zip(
                infos["_episode"], infos["episode"]["r"], infos["episode"]["l"]
            ):
                if is_final:
                    returns.append(r)
                    lengths.append(l)
    return global_step, list(returns), list(lengths)


def update(
    agent: Agent,
    optimizer: optim.Optimizer,
    obs: Batch,
    actions: Tensor,
    logprobs: Tensor,
    advantages: Tensor,
    returns: Tensor,
    values: Tensor,
    clip_coef: float,
    norm_adv: bool,
    clip_vloss: bool,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
):
    newlogprob, entropy, newvalue = agent.evaluate_action_and_value(
        actions,
        obs.x,
        obs.edge_index,
        obs.edge_attr,
        obs.batch,
        # torch.ones_like(obs.action_mask),
        # torch.ones_like(obs.node_mask),
    )
    assert not newlogprob.isinf().any()
    assert newlogprob.dim() == 1
    assert entropy.dim() == 1
    assert newvalue.dim() == 2
    logratio = newlogprob - logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfracs = [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

    if norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss
    pg_loss1 = torch.mul(-advantages, ratio)
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue = newvalue.view(-1)
    if clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = values + torch.clamp(
            newvalue - values,
            -clip_coef,
            clip_coef,
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

    assert not torch.isnan(loss).any(), loss

    optimizer.zero_grad()
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(
        agent.parameters(), max_grad_norm, error_if_nonfinite=True
    )
    optimizer.step()
    return (
        loss,
        pg_loss,
        v_loss,
        entropy_loss,
        old_approx_kl,
        approx_kl,
        grad_norm,
        clipfracs,
    )


def main(run_name: str, args: Args, agent_config: dict[str, Any]):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    kg_wrapper.register_env()
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ],
    )

    agent = Agent(envs, **agent_config).to(device)

    if args.track:
        wandb.watch(agent, log_freq=10, log="all")

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs: deque[Data] = deque()
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = dict_to_data(next_obs, args.num_envs)
    next_done = torch.zeros(args.num_envs).to(device)

    pbar = tqdm(total=args.num_iterations)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        global_step, rollout_returns, rollout_lengths = rollout(
            agent,
            envs,
            next_obs,
            next_done,
            obs,
            dones,
            rewards,
            actions,
            logprobs,
            values,
            args.num_steps,
            args.num_envs,
            device,
            global_step,
        )

        r = np.mean(rollout_returns) if rollout_returns else 0
        l = np.mean(rollout_lengths) if rollout_lengths else 0

        # bootstrap value if not done
        with torch.no_grad():
            next_obs_batch = create_batch(next_obs)
            next_value = agent.get_value(
                next_obs_batch.x,
                next_obs_batch.edge_index,
                next_obs_batch.edge_attr,
                next_obs_batch.batch,
                # next_obs_batch.action_mask,
                # next_obs_batch.node_mask,
            ).reshape(1, -1)
            advantages, returns = gae(
                rewards,
                dones,
                values,
                next_value,
                next_done,
                args.num_steps,
                args.gamma,
                args.gae_lambda,
                device,
            )

        # flatten the batch
        b_obs = Batch.from_data_list(obs)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                minibatch = Batch.from_data_list(b_obs.index_select(mb_inds))
                (
                    loss,
                    pg_loss,
                    v_loss,
                    entropy_loss,
                    old_approx_kl,
                    approx_kl,
                    grad_norm,
                    clipfracs,
                ) = update(
                    agent,
                    optimizer,
                    minibatch,
                    b_actions.long()[mb_inds],
                    b_logprobs[mb_inds],
                    b_advantages[mb_inds],
                    b_returns[mb_inds],
                    b_values[mb_inds],
                    args.clip_coef,
                    args.norm_adv,
                    args.clip_vloss,
                    args.ent_coef,
                    args.vf_coef,
                    args.max_grad_norm,
                )
                clipfracs += clipfracs

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        mlflow.log_metric(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        save_agent(envs, agent, agent_config, f"{run_name}.pth")
        mlflow.log_artifact(f"{run_name}.pth")
        pbar.update(1)
        pbar.set_description(
            f"R:{r:.2f} | L:{l:.2f} | ENT:{entropy_loss:.2f} | EXPL_VARIANCE:{explained_var:.2f}"
        )

        mlflow.log_metric("rollout/mean_reward", rewards.mean().item(), global_step)
        mlflow.log_metric("rollout/mean_episodic_return", r, global_step)
        mlflow.log_metric("rollout/mean_episodic_length", l, global_step)
        mlflow.log_metric("rollout/advantages", b_advantages.mean().item(), global_step)
        mlflow.log_metric("rollout/returns", b_returns.mean().item(), global_step)
        mlflow.log_metric("rollout/values", b_values.mean().item(), global_step)
        mlflow.log_metric("losses/total_loss", loss.item(), global_step)
        mlflow.log_metric("losses/grad_norm", grad_norm, global_step)
        mlflow.log_metric("losses/value_loss", v_loss.item(), global_step)
        mlflow.log_metric("losses/policy_loss", pg_loss.item(), global_step)
        mlflow.log_metric("losses/entropy", entropy_loss.item(), global_step)
        mlflow.log_metric("losses/old_approx_kl", old_approx_kl.item(), global_step)
        mlflow.log_metric("losses/approx_kl", approx_kl.item(), global_step)
        mlflow.log_metric("losses/clipfrac", np.mean(clipfracs), global_step)
        mlflow.log_metric("losses/explained_variance", explained_var, global_step)
        mlflow.log_metric(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    envs.close()


def setup():
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    args = tyro.cli(Args)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"

    agent_config = {
        "layers": 2,
        "embedding_dim": 256,
        "aggregation": "sum",
        "activation": nn.Tanh(),
    }

    config = vars(args) | agent_config

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=config,
            name=run_name,
            save_code=True,
        )

    try:
        mlflow.create_experiment(run_name)
    except Exception:
        pass

    mlflow.set_experiment(run_name)

    with mlflow.start_run():
        mlflow.log_params(config)
        mlflow.log_artifact(__file__)
        mlflow.log_artifact("kg_gnn.py")
        main(run_name, args, agent_config)


if __name__ == "__main__":
    setup()
