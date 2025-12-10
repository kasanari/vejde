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
import tyro

import mlflow

from gnn_policy.functional import segment_sum


from regawa.data import (
    heterostatedata,
    heterostatedata_to_tensors,
)

from regawa.policy.q_agent.gnn_q_agent import GraphQAgent
from regawa.rl.graph_buffer import ReplayBuffer
from regawa import agent_from_env
from regawa import GNNParams


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




if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)
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

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(
            1 / torch.tensor(envs.single_action_space.n)
        )
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
                    _, next_state_log_pi, next_state_action_probs = actor.get_action(
                        data.next_observations
                    )
                    qf1_next_target = qf1_target(data.next_observations)
                    qf2_next_target = qf2_target(data.next_observations)
                    # we can use the action probabilities instead of MC sampling to estimate the expectation
                    min_qf_next_target = next_state_action_probs * (
                        torch.min(qf1_next_target, qf2_next_target)
                        - alpha * next_state_log_pi
                    )
                    # adapt Q-target for discrete Q-function
                    min_qf_next_target = min_qf_next_target.sum(axis=1)
                    next_q_value = data.rewards.flatten() + (
                        1 - data.dones.flatten()
                    ) * args.gamma * (min_qf_next_target)

                # use Q-values only for the taken actions
                qf1_values = qf1(data.observations)
                qf2_values = qf2(data.observations)
                qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
                qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # ACTOR training
                _, log_pi, action_probs = actor.get_action(data.observations)
                with torch.no_grad():
                    qf1_values = qf1(data.observations)
                    qf2_values = qf2(data.observations)
                    min_qf_values = torch.min(qf1_values, qf2_values)
                # no need for reparameterization, the expectation can be calculated for discrete actions
                actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    # re-use action probabilities for temperature loss
                    alpha_loss = (
                        action_probs.detach()
                        * (-log_alpha.exp() * (log_pi + target_entropy).detach())
                    ).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            if global_step % 100 == 0:
                mlflow.log_param(
                    "agent/qf1_values", qf1_a_values.mean().item(), global_step
                )
                mlflow.log_param(
                    "agent/qf2_values", qf2_a_values.mean().item(), global_step
                )
                mlflow.log_param("losses/qf1_loss", qf1_loss.item(), global_step)
                mlflow.log_param("losses/qf2_loss", qf2_loss.item(), global_step)
                mlflow.log_param("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                mlflow.log_param("losses/actor_loss", actor_loss.item(), global_step)
                mlflow.log_param("losses/alpha", alpha, global_step)
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
                mlflow.log_param(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    mlflow.log_param(
                        "losses/alpha_loss", alpha_loss.item(), global_step
                    )

    envs.close()


if __name__ == "__main__":
    args: SACArgs = tyro.cli(SACArgs)
    train(args)
