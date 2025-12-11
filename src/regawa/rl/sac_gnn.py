# type: ignore
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
import os
import random
import time
from typing import NamedTuple

import gymnasium as gym
from gymnasium.spaces import Dict, MultiDiscrete
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from tqdm import tqdm
from regawa.rl.sac import (
    sac_action_then_node_entropy,
    sac_action_then_node_value_estimate,
    sac_action_then_node_policy_loss,
)
import tyro
from functools import partial

import mlflow

from gnn_policy.functional import (
    segment_sum,
    segmented_gather,
    data_splits_and_starts,
    node_logits_given_action,
)


from regawa.data import (
    heterostatedata,
    heterostatedata_to_tensors,
    HeteroBatchData,
)

from regawa.policy.q_agent.gnn_q_agent import GraphQAgent
from regawa.rl.graph_buffer import ReplayBuffer
from regawa import agent_from_env
from regawa import GNNParams
from regawa.policy.q_agent.q_value import QValue

class SACArgs(NamedTuple):
    env_id: str
    """the id of the environment"""
    agent_class: str
    """the agent class to use"""
    agent_config: GNNParams
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    mlflow_tracking_uri: str = ""
    """the tracking uri for mlflow. If empty, mlflow will log locally"""
    multiprocess: bool = False
    """whether to use multiprocessed envs"""
    num_envs: int = 8
    """the number of parallel game environments"""
    # Algorithm specific arguments
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    buffer_size: int = int(100)
    """the replay memory buffer size"""  # smaller than in original paper but evaluation is done only for 100k steps anyway
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """target smoothing coefficient (default: 1)"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    learning_starts: int = 0
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    update_frequency: int = 4
    """the frequency of training updates"""
    target_network_frequency: int = 8000
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    target_entropy_scale: float = 0.89
    """coefficient for scaling the autotune entropy target"""
    weight_decay: float = 0.0
    """weight decay for optimizers"""
    debug: bool = False
    """whether to run in debug mode"""


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


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DoubleQNetwork(nn.Module):
    def __init__(self, a1: GraphQAgent, a2: GraphQAgent):
        super(DoubleQNetwork, self).__init__()
        self.q1 = a1
        self.q2 = a2

    def forward(
        self,
        obs: HeteroBatchData,
    ) -> tuple[QValue, QValue]:
        return (self.q1(obs), self.q2(obs))

    def min(self, obs: HeteroBatchData) -> QValue:
        q1_values, q2_values = self.forward(obs)
        return q1_values.min(q2_values)


def train(args: SACArgs) -> None:
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        mlflow.enable_system_metrics_logging()
        mlflow.set_tracking_uri(uri=args.mlflow_tracking_uri)
        try:
            mlflow.create_experiment(run_name)
        except mlflow.MlflowException:
            pass
        mlflow.set_experiment(run_name)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    pbar = tqdm(range(args.total_timesteps), dynamic_ncols=True)

    # env setup
    envs: gym.vector.VectorEnv = (
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

    actor = agent_from_env("GraphAgent", envs, args.agent_config, device)
    qf1 = agent_from_env("GraphQAgent", envs, args.agent_config, device)
    qf2 = agent_from_env("GraphQAgent", envs, args.agent_config, device)
    q_net = DoubleQNetwork(qf1, qf2).to(device)
    qf1_target = agent_from_env("GraphQAgent", envs, args.agent_config, device)
    qf2_target = agent_from_env("GraphQAgent", envs, args.agent_config, device)
    target_net = DoubleQNetwork(qf1_target, qf2_target).to(device)
    q_net.load_state_dict(target_net.state_dict())
    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    q_optimizer = optim.Adam(q_net.parameters(), lr=args.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr, eps=1e-4)

    # Automatic entropy tuning
    if args.autotune:
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        seed=args.seed,
        n_envs=args.num_envs,
    )
    start_time = time.time()
    returns: deque[float] = deque()
    lengths: deque[int] = deque()

    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions, *_ = actor.sample(
                heterostatedata_to_tensors(heterostatedata(obs), device=device)
            )
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "episode" in infos:
            for f, r, length in zip(
                infos["_episode"], infos["episode"]["r"], infos["episode"]["l"]
            ):
                if f:
                    returns.append(r)
                    lengths.append(length)

        dones = np.logical_or(terminations, truncations)
        rb.add(obs, next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.update_frequency == 0:
                data = rb.sample(args.batch_size)
                # CRITIC training
                with torch.no_grad():
                    next_obs_as_tensor = heterostatedata_to_tensors(
                        data.next_observations
                    )
                    x = actor.sample(next_obs_as_tensor)

                    target_q_next = target_net.min(next_obs_as_tensor)
                    log_p_a = torch.log(x.p_a1)
                    log_p_n__a = torch.log(x.p_a2 + 1e-10)
                    # we can use the action probabilities instead of MC sampling to estimate the expectation
                    vf_target = sac_action_then_node_value_estimate(
                        x.p_a2,
                        target_q_next.q2.values,
                        x.p_a1,
                        log_p_a,
                        log_p_n__a,
                        alpha,
                        partial(
                            segment_sum,
                            index=target_q_next.q2.indices,
                            num_segments=data.next_observations.n_graphs,
                        ),
                    )
                    next_q_value = (
                        data.rewards.flatten()
                        + (1 - data.dones.flatten()) * args.gamma * vf_target
                    )

                # use Q-values only for the taken actions
                obs_as_tensor = heterostatedata_to_tensors(
                    data.observations, device=device
                )
                qs = q_net.forward(obs_as_tensor)

                _, data_starts = data_splits_and_starts(obs_as_tensor.n_factor)

                def q_loss(q_values: QValue) -> torch.Tensor:
                    a = data.actions.long()
                    # q values for all nodes given action a
                    q_action = node_logits_given_action(
                        q_values.q2.values, a[:, 0], q_values.q2.indices
                    )
                    # one q value per graph, for the taken node action
                    q_action = segmented_gather(q_action, a[:, 1], data_starts)
                    return F.mse_loss(q_action, next_q_value)

                qf_loss = torch.sum(torch.stack([q_loss(q) for q in qs]))

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # ACTOR training
                x = actor.sample(obs_as_tensor)
                with torch.no_grad():
                    qf_values = q_net.min(obs_as_tensor)
                # no need for reparameterization, the expectation can be calculated for discrete actions
                actor_loss = sac_action_then_node_policy_loss(
                    x.p_a2,
                    qf_values.q2.values,
                    x.p_a1,
                    torch.log(x.p_a1),
                    torch.log(x.p_a2 + 1e-10),
                    alpha,
                    partial(
                        segment_sum,
                        index=qf_values.q2.indices,
                        num_segments=obs_as_tensor.n_graphs,
                    ),
                )

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    # re-use action probabilities for temperature loss
                    batch_idx = qf_values.q2.indices
                    target_p = -args.target_entropy_scale * torch.log(
                        1 / obs_as_tensor.boolean.factor.n_factor
                    )
                    target_p = target_p[batch_idx].unsqueeze(-1)
                    target_a = -args.target_entropy_scale * torch.log(
                        1 / torch.tensor(envs.single_action_space.nvec[0])
                    )

                    alpha_loss = sac_action_then_node_entropy(
                        x.p_a2,
                        x.p_a1,
                        torch.log(x.p_a1),
                        torch.log(x.p_a2 + 1e-10),
                        log_alpha,
                        target_a,
                        target_p,
                        partial(
                            segment_sum,
                            index=qf_values.q2.indices,
                            num_segments=obs_as_tensor.n_graphs,
                        ),
                    )

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(
                    q_net.parameters(), target_net.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            if global_step % 100 == 0:
                # mlflow.log_param(
                #     "agent/qf1_values", qf1_a_values.mean().item(), global_step
                # )
                # mlflow.log_param(
                #     "agent/qf2_values", qf2_a_values.mean().item(), global_step
                # )

                mlflow.log_metric("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                mlflow.log_metric("losses/actor_loss", actor_loss.item(), global_step)
                mlflow.log_metric("losses/alpha", alpha, global_step)
                # tw("SPS:", int(global_step / (time.time() - start_time)))
                pbar.update(100)
                avg_episodic_return = (
                    sum(returns) / len(returns) if len(returns) > 0 else 0.0
                )
                avg_episodic_length = (
                    sum(lengths) / len(lengths) if len(lengths) > 0 else 0.0
                )
                pbar.set_description(
                    f"Step: {global_step}, R: {avg_episodic_return.item():.2f}, L: {avg_episodic_length.item():.2f}"
                )
                mlflow.log_metric(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                mlflow.log_metric(
                    "charts/avg_episodic_return",
                    avg_episodic_return.item(),
                    global_step,
                )
                mlflow.log_metric(
                    "charts/avg_episodic_length",
                    avg_episodic_length.item(),
                    global_step,
                )
                if args.autotune:
                    mlflow.log_metric(
                        "losses/alpha_loss", alpha_loss.item(), global_step
                    )

    envs.close()


if __name__ == "__main__":
    args: SACArgs = tyro.cli(SACArgs)
    train(args)
