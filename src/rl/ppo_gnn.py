# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
from collections import deque
import os
import random
import time
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro


from gymnasium.spaces import Dict, MultiDiscrete
from torch import Tensor
import wandb
from tqdm import tqdm
from typing import Any

from torch_geometric.data import Data

from gnn.data import (
    BipartiteData,
    batched_dict_to_data,
    statedata_from_obs,
)
from rddl import register_env

from gnn import GraphAgent, Config, ActionMode, StateData
import mlflow
import mlflow.pytorch


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
    env_id: str = "rddl/conditional_bandit.rddl"  # "Elevators_MDP_ippc2011"
    """the id of the environment"""
    instance: str | int = (
        "rddl/conditional_bandit_i0.rddl"  # "rddl/elevators_mdp__ippc11.rddl"
    )
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
    gae_lambda: float = 1.0
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 40
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
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


def make_env(
    env_id: str,
    domain: str,
    instance: str | int,
    idx: int,
    capture_video: bool,
    run_name: str,
):
    def thunk() -> gym.Env[Dict, MultiDiscrete]:
        env: gym.Env[Dict, MultiDiscrete] = gym.make(
            env_id,
            domain=domain,
            instance=instance,
            # add_inverse_relations=False,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


class Agent(nn.Module):
    def __init__(
        self,
        config: Config,
        **kwargs: dict[str, Any],
    ):
        super().__init__()  # type: ignore

        self.agent = GraphAgent(
            config,
        )

    def get_value(
        self,
        s: StateData,
    ):
        value = self.agent.value(s)
        return value

    def sample_action_and_value(self, s: StateData):
        action, logprob, entropy, value = self.agent.sample(
            s,
        )
        return action, logprob, entropy, value

    def evaluate_action_and_value(
        self,
        action: Tensor,
        s: StateData,
    ):
        # num_graphs = batch_idx.max() + 1
        # action_mask = action_mask.reshape(num_graphs, -1)
        logprob, entropy, value = self.agent.forward(
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


@torch.inference_mode()
def rollout(
    agent: Agent,
    envs: gym.vector.SyncVectorEnv,
    next_obs: list[BipartiteData],
    next_done: Tensor,
    dones: Tensor,
    rewards: Tensor,
    actions: Tensor,
    logprobs: Tensor,
    values: Tensor,
    num_steps: int,
    num_envs: int,
    device: torch.device | str,
    global_step: int,
) -> tuple[list[Data], int, list[float], list[int]]:
    returns: deque[float] = deque()
    lengths: deque[int] = deque()
    obs: deque[BipartiteData] = deque()
    for step in range(0, num_steps):
        global_step += num_envs
        obs.extend(next_obs)
        dones[step] = next_done

        # ALGO LOGIC: action logic
        s = statedata_from_obs(next_obs)
        action, logprob, _, value = agent.sample_action_and_value(
            s
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
        next_obs_dict, reward, terminations, truncations, infos = envs.step(  # type: ignore
            action.cpu().numpy()  # type: ignore
        )
        next_done = np.logical_or(terminations, truncations)
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = (
            batched_dict_to_data(next_obs_dict, num_envs),  # type: ignore
            torch.Tensor(next_done).to(device),
        )

        if "episode" in infos:
            for is_final, r, l in zip(
                infos["_episode"], infos["episode"]["r"], infos["episode"]["l"]
            ):
                if is_final:
                    returns.append(r)
                    lengths.append(l)
    return list(obs), global_step, list(returns), list(lengths)


def update(
    agent: Agent,
    optimizer: optim.Optimizer,
    s: StateData,
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
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, list[float]]:
    newlogprob, entropy, newvalue = agent.evaluate_action_and_value(
        actions,
        s,
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
    loss.backward()  # type: ignore
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


def main(
    envs: gym.vector.SyncVectorEnv,
    run_name: str,
    args: Args,
    agent_config: Config,
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
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape  # type: ignore
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)  # type: ignore
    next_obs = batched_dict_to_data(next_obs, args.num_envs)  # type: ignore
    next_done = torch.zeros(args.num_envs).to(device)
    approx_kl = 0.0
    entropy_loss = 0.0

    pbar = tqdm(total=args.num_iterations)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        obs, global_step, rollout_returns, rollout_lengths = rollout(
            agent,
            envs,
            next_obs,
            next_done,
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

        r = np.mean(rollout_returns) if rollout_returns else 0.0
        length = np.mean(rollout_lengths) if rollout_lengths else 0.0

        # bootstrap value if not done
        with torch.no_grad():
            next_obs_batch = statedata_from_obs(next_obs)
            next_value = agent.get_value(next_obs_batch).reshape(1, -1)
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
        # b_obs = Batch.from_data_list(obs)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)  # type: ignore
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        batch_size = args.batch_size
        minibatch_size = args.minibatch_size
        update_epochs = args.update_epochs
        b_inds = np.arange(batch_size)
        clipfracs: list[float] = []
        for _ in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                minibatch = statedata_from_obs([obs[i] for i in mb_inds])
                (
                    loss,
                    pg_loss,
                    v_loss,
                    entropy_loss,
                    old_approx_kl,
                    approx_kl,
                    grad_norm,
                    clipfrac,
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
                clipfracs += clipfrac

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()  # type: ignore
        var_y = np.var(y_true)  # type: ignore
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        mlflow.log_metric(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        agent.agent.save_agent(f"{run_name}.pth")
        mlflow.log_artifact(f"{run_name}.pth")
        pbar.update(1)
        pbar.set_description(
            f"R:{r:.2f} | L:{length:.2f} | ENT:{entropy_loss:.2f} | EXPL_VARIANCE:{explained_var:.2f}"
        )

        mlflow.log_metric("rollout/mean_reward", rewards.mean().item(), global_step)
        mlflow.log_metric("rollout/mean_episodic_return", r, global_step)  # type: ignore
        mlflow.log_metric("rollout/mean_episodic_length", length, global_step)  # type: ignore
        mlflow.log_metric("rollout/advantages", b_advantages.mean().item(), global_step)
        mlflow.log_metric("rollout/returns", b_returns.mean().item(), global_step)
        mlflow.log_metric("rollout/values", b_values.mean().item(), global_step)
        mlflow.log_metric("losses/total_loss", loss.item(), global_step)  # type: ignore
        mlflow.log_metric("losses/grad_norm", grad_norm, global_step)  # type: ignore
        mlflow.log_metric("losses/value_loss", v_loss.item(), global_step)  # type: ignore
        mlflow.log_metric("losses/policy_loss", pg_loss.item(), global_step)  # type: ignore
        mlflow.log_metric("losses/entropy", entropy_loss.item(), global_step)  # type: ignore
        mlflow.log_metric("losses/old_approx_kl", old_approx_kl.item(), global_step)  # type: ignore
        mlflow.log_metric("losses/approx_kl", approx_kl.item(), global_step)  # type: ignore
        mlflow.log_metric("losses/clipfrac", np.mean(clipfracs), global_step)  # type: ignore
        mlflow.log_metric("losses/explained_variance", explained_var, global_step)  # type: ignore
        mlflow.log_metric(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    envs.close()


def setup():
    print("Attempting to connect to mlflow...")
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    print("Connected to mlflow.")
    args = tyro.cli(Args)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"

    env_id = register_env()
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id, args.env_id, args.instance, i, args.capture_video, run_name
            )
            for i in range(args.num_envs)
        ],
    )

    n_types = int(envs.single_observation_space["factor"].feature_space.n)  # type: ignore
    n_relations = int(envs.single_observation_space["var_type"].feature_space.n)  # type: ignore
    n_actions = int(envs.single_action_space.nvec[0])  # type: ignore

    agent_config = Config(
        n_types,
        n_relations,
        n_actions,
        layers=4,
        embedding_dim=16,
        activation=nn.LeakyReLU(),
        aggregation="sum",
        action_mode=ActionMode.ACTION_THEN_NODE,
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

    with mlflow.start_run():
        mlflow.log_params(logged_config)
        mlflow.log_artifact(__file__)
        mlflow.log_artifact("kg_gnn.py")
        main(envs, run_name, args, agent_config)


if __name__ == "__main__":
    setup()
