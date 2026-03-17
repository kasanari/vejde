# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
import contextlib
import os
import random
import time
from collections import deque
from collections.abc import Iterable
from typing import NamedTuple

import gymnasium as gym
import mlflow  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from gnn_policy.functional import (
    data_splits_and_starts,
    node_logits_given_action,
    segmented_gather,
)
from gymnasium.spaces import Dict, MultiDiscrete
from numpy.typing import NDArray
from torch import nn, optim
from tqdm import tqdm

from regawa import GNNParams, agent_from_env
from regawa.data import (
    HeteroBatchData,
    heterostatedata,
    heterostatedata_to_tensors,
)
from regawa.data.obs import HeteroObsData
from regawa.data.torch import SparseTensor, TorchHeteroBatchData
from regawa.policy import GraphAgentInterface
from regawa.policy.q_agent.gnn_q_agent import GraphQAgent
from regawa.policy.q_agent.q_value import QValue
from regawa.policy.types import PolicyOutput
from regawa.rl.graph_buffer import ReplayBuffer, ReplayBufferSamples
from regawa.rl.sac import (
    sac_action_then_node_entropy,
    sac_action_then_node_policy_loss,
    sac_action_then_node_value_estimate,
)


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
    buffer_size: int = 100
    """the replay memory buffer size"""
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
    target_entropy_scale_a1: float = 0.89
    target_entropy_scale_a2: float = 0.3
    """coefficient for scaling the autotune entropy target"""
    weight_decay: float = 0.0
    """weight decay for optimizers"""
    debug: bool = False
    """whether to run in debug mode"""
    logging_interval: int = 100


def make_env(
    env_id: str,
):
    def thunk() -> gym.Env[Dict, MultiDiscrete]:
        env: gym.Env[Dict, MultiDiscrete] = gym.make(  # type: ignore
            env_id,
        )
        return gym.wrappers.RecordEpisodeStatistics(env)

    return thunk


def layer_init(layer: nn.Linear, bias_const: float = 0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DoubleQNetwork(nn.Module):
    def __init__(self, a1: GraphQAgent, a2: GraphQAgent):
        super().__init__()  # type: ignore
        self.q1 = a1
        self.q2 = a2

    def forward(
        self,
        obs: TorchHeteroBatchData,
    ) -> tuple[QValue, QValue]:
        return (self.q1(obs), self.q2(obs))

    def min(self, obs: TorchHeteroBatchData) -> QValue:
        q1_values, q2_values = self.forward(obs)
        return q1_values.min(q2_values)


@torch.no_grad()  # type: ignore
def get_next_q_value(
    actor: GraphAgentInterface,
    target_net: DoubleQNetwork,
    data: ReplayBufferSamples,
    alpha: float,
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    next_obs_as_tensor = heterostatedata_to_tensors(data.next_observations, device)
    x = actor.sample(next_obs_as_tensor)

    target_q_next = target_net.min(next_obs_as_tensor)
    assert isinstance(x.p_a1, torch.Tensor)
    assert isinstance(x.p_a2, SparseTensor)
    assert isinstance(target_q_next.q2, SparseTensor)
    assert isinstance(target_q_next.q1, torch.Tensor)
    log_p_a = torch.log(x.p_a1)
    log_p_n__a = x.p_a2.map(lambda x: torch.where(x == 0, 1, x)).map(torch.log)
    vf_target = sac_action_then_node_value_estimate(
        x.p_a2,
        target_q_next.q2,
        x.p_a1,
        target_q_next.q1,
        log_p_a,
        log_p_n__a,
        alpha,
        n_graphs=int(data.next_observations.n_graphs),
    )
    return data.rewards.flatten() + (1 - data.dones.flatten()) * gamma * vf_target


def update_q_net(
    obs: TorchHeteroBatchData,
    actions: torch.Tensor,
    q_net: DoubleQNetwork,
    q_optimizer: optim.Optimizer,
    next_q_value: torch.Tensor,
    _device: torch.device,
):
    # use Q-values only for the taken actions

    qs = q_net.forward(obs)

    _, data_starts = data_splits_and_starts(obs.n_factor)

    def q_loss(q_values: QValue) -> torch.Tensor:
        assert isinstance(q_values.q2, SparseTensor)
        # q values for all nodes given action a
        q_action = node_logits_given_action(
            q_values.q2.values, actions[:, 0], q_values.q2.indices
        )
        # one q value per graph, for the taken node action
        q_action = segmented_gather(q_action, actions[:, 1], data_starts)
        return F.mse_loss(q_action, next_q_value)

    qf_loss = torch.sum(torch.stack([q_loss(q) for q in qs]))

    q_optimizer.zero_grad()
    qf_loss.backward()  # type: ignore
    q_optimizer.step()
    return qf_loss


def update_actor(
    actor: GraphAgentInterface,
    q_net: DoubleQNetwork,
    obs_as_tensor: TorchHeteroBatchData,
    alpha: float,
    actor_optimizer: optim.Optimizer,
):
    x = actor.sample(obs_as_tensor)
    with torch.no_grad():
        qf_values = q_net.min(obs_as_tensor)
    # no need for reparameterization, the expectation can be calculated for discrete actions
    assert isinstance(x.p_a1, torch.Tensor)
    assert isinstance(x.p_a2, SparseTensor)
    assert isinstance(qf_values.q2, SparseTensor)
    assert isinstance(qf_values.q1, torch.Tensor)
    assert isinstance(x.p_a1, torch.Tensor)

    p_a2 = x.p_a2.map(lambda x: torch.where(x == 0, 1, x)).map(torch.log)
    actor_loss = sac_action_then_node_policy_loss(
        x.p_a2,
        qf_values.q2,
        x.p_a1,
        qf_values.q1,
        torch.log(x.p_a1),
        p_a2,
        alpha,
        num_graphs=int(obs_as_tensor.n_graphs),
    )

    actor_optimizer.zero_grad()
    actor_loss.backward()  # type: ignore
    actor_optimizer.step()
    return x, qf_values, actor_loss


def update_alpha(
    policy_output: PolicyOutput,
    log_alpha: torch.Tensor,
    a_optimizer: optim.Optimizer,
    target_a: torch.Tensor,
    n_factor: torch.Tensor,
    n_graphs: int,
    target_entropy_scale: float,
) -> tuple[float, torch.Tensor]:
    # re-use action probabilities for temperature loss
    assert isinstance(policy_output.p_a2, SparseTensor)
    batch_index = policy_output.p_a2.indices
    target_p = -target_entropy_scale * torch.log(1 / n_factor)
    target_p = target_p[batch_index].unsqueeze(-1)
    assert isinstance(policy_output.p_a1, torch.Tensor)
    assert isinstance(log_alpha, torch.Tensor)
    assert isinstance(batch_index, torch.Tensor)
    p_a2 = policy_output.p_a2.map(lambda x: torch.where(x == 0, 1, x)).map(torch.log)
    alpha_loss = sac_action_then_node_entropy(
        policy_output.p_a2,
        policy_output.p_a1,
        torch.log(policy_output.p_a1),
        p_a2,
        log_alpha,
        target_a,
        target_p,
        n_graphs,
    )

    a_optimizer.zero_grad()
    alpha_loss.backward()  # type: ignore
    a_optimizer.step()
    alpha = log_alpha.exp().item()
    return alpha, alpha_loss


class UpdateModelsOutput(NamedTuple):
    new_alpha: float
    alpha_loss: torch.Tensor
    actor_loss: torch.Tensor
    qf_loss: torch.Tensor
    next_q_value: float
    q1_mean: float
    q2_mean: float


def update_models(
    data: ReplayBufferSamples,
    actor: GraphAgentInterface,
    log_alpha: torch.Tensor | None,
    target_q_net: DoubleQNetwork,
    q_net: DoubleQNetwork,
    alpha: float,
    device: torch.device,
    target_a: torch.Tensor,
    q_optimizer: optim.Optimizer,
    actor_optimizer: optim.Optimizer,
    alpha_optimizer: optim.Optimizer | None,
    gamma: float,
    target_entropy_scale: float,
) -> UpdateModelsOutput:
    obs_as_tensor = heterostatedata_to_tensors(data.observations, device=device)
    next_q_value = get_next_q_value(actor, target_q_net, data, alpha, gamma, device)

    # CRITIC training
    qf_loss = update_q_net(
        obs_as_tensor, data.actions.long(), q_net, q_optimizer, next_q_value, device
    )

    # ACTOR training
    policy_output, qf_values, actor_loss = update_actor(
        actor,
        q_net,
        obs_as_tensor,
        alpha,
        actor_optimizer,
    )

    # ALPHA training
    if log_alpha is not None and alpha_optimizer is not None:
        new_alpha, alpha_loss = update_alpha(
            policy_output,
            log_alpha,
            alpha_optimizer,
            target_a,
            obs_as_tensor.n_factor,
            obs_as_tensor.n_graphs,
            target_entropy_scale,
        )
    else:
        new_alpha = alpha
        alpha_loss = torch.tensor(0.0)

    average_next_q_value = next_q_value.mean().item()
    q1_mean = qf_values.q1.mean().item()  # type: ignore
    q2_mean = qf_values.q2.segment_mean().mean().item()  # type: ignore
    return UpdateModelsOutput(
        new_alpha,
        alpha_loss,
        actor_loss,
        qf_loss,
        average_next_q_value,
        q1_mean,
        q2_mean,
    )


def log_to_mlflow(
    global_step: int,
    update_data: UpdateModelsOutput,
    alpha: float,
    avg_episodic_return: float,
    avg_episodic_length: float,
    start_time: float,
    args: SACArgs,
):
    # mlflow.log_param(
    #     "agent/qf1_values", qf1_a_values.mean().item(), global_step
    # )
    # mlflow.log_param(
    #     "agent/qf2_values", qf2_a_values.mean().item(), global_step
    # )

    mlflow.log_metric("losses/qf_loss", update_data.qf_loss.item() / 2.0, global_step)
    mlflow.log_metric("losses/actor_loss", update_data.actor_loss.item(), global_step)
    mlflow.log_metric("losses/alpha", alpha, global_step)
    # tw("SPS:", int(global_step / (time.time() - start_time)))

    mlflow.log_metric("stats/next_q_value", update_data.next_q_value, global_step)
    mlflow.log_metric("stats/q1_mean", update_data.q1_mean, global_step)
    mlflow.log_metric("stats/q2_mean", update_data.q2_mean, global_step)

    mlflow.log_metric(
        "charts/SPS",
        int(global_step / (time.time() - start_time)),
        global_step,
    )
    mlflow.log_metric(
        "charts/avg_episodic_return",
        avg_episodic_return,
        global_step,
    )
    mlflow.log_metric(
        "charts/avg_episodic_length",
        avg_episodic_length,
        global_step,
    )
    if args.autotune:
        mlflow.log_metric(
            "losses/alpha_loss", update_data.alpha_loss.item(), global_step
        )


def sample_action(obs: HeteroObsData, rng: np.random.Generator) -> np.ndarray:
    a1_mask = obs.bool.action_masks.action_arity_mask
    a2_mask = obs.float.action_masks.action_type_mask
    mask = a1_mask & a2_mask
    action = np.flip(np.stack(np.where(mask)).T)
    idx = rng.choice(action.shape[0])
    return action[idx]  # type: ignore


def step_fn(
    start_time: float,
    actor: GraphAgentInterface,
    q_net: DoubleQNetwork,
    log_alpha: torch.Tensor | None,
    target_net: DoubleQNetwork,
    a_optimizer: optim.Optimizer | None,
    q_optimizer: optim.Optimizer,
    device: torch.device,
    target_a: torch.Tensor,
    actor_optimizer: optim.Optimizer,
    envs: gym.vector.VectorEnv[HeteroBatchData, NDArray[np.int64], NDArray[np.int64]],
    pbar: tqdm,
    args: SACArgs,
    rng: np.random.Generator,
):
    def step(
        obs: Iterable[HeteroObsData],
        rb: ReplayBuffer,
        global_step: int,
        returns: deque[float],
        lengths: deque[int],
        alpha: float,
    ):
        if global_step < args.learning_starts:
            actions = np.array([sample_action(o, rng) for o in obs])
        else:
            actions, *_ = actor.sample(
                heterostatedata_to_tensors(heterostatedata(obs), device=device)
            )
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

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

        dones = np.logical_or(terminations, truncations)
        rb.add(obs, next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs  # type: ignore

        update_data = UpdateModelsOutput(
            new_alpha=alpha,
            alpha_loss=torch.tensor(0.0),
            actor_loss=torch.tensor(0.0),
            qf_loss=torch.tensor(0.0),
            next_q_value=0.0,
            q1_mean=0.0,
            q2_mean=0.0,
        )

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.update_frequency == 0:
                data = rb.sample(args.batch_size)
                update_data = update_models(
                    data,
                    actor,
                    log_alpha,
                    target_net,
                    q_net,
                    alpha,
                    device,
                    target_a,
                    q_optimizer,
                    actor_optimizer,
                    a_optimizer,
                    args.gamma,
                    args.target_entropy_scale_a2,
                )
                alpha = update_data.new_alpha

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(
                    q_net.parameters(), target_net.parameters(), strict=False
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            if global_step % args.logging_interval == 0 and global_step > 0:
                pbar.update(args.logging_interval)
                avg_episodic_return = (
                    sum(returns) / len(returns) if len(returns) > 0 else 0.0
                )
                avg_episodic_length = (
                    sum(lengths) / len(lengths) if len(lengths) > 0 else 0.0
                )

                log_to_mlflow(
                    global_step,
                    update_data,
                    alpha,
                    avg_episodic_return,
                    avg_episodic_length,
                    start_time,
                    args,
                )
                pbar.set_description(
                    f"Step: {global_step}, R: {avg_episodic_return:.2f}, L: {avg_episodic_length:.2f}"
                )
        return obs, alpha, returns, lengths

    return step


def train(args: SACArgs) -> GraphAgentInterface:
    run_name = f"{args.env_id}__sac"
    run_name = run_name + "__debug" if args.debug else run_name
    if args.track:
        mlflow.enable_system_metrics_logging()
        mlflow.set_tracking_uri(uri=args.mlflow_tracking_uri)
        with contextlib.suppress(mlflow.MlflowException):
            mlflow.create_experiment(run_name)
        mlflow.set_experiment(run_name)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    _rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)  # type: ignore
    torch.backends.cudnn.deterministic = args.torch_deterministic

    mlflow.log_params(args._asdict())

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    pbar = tqdm(range(args.total_timesteps), dynamic_ncols=True)

    # env setup
    envs: gym.vector.VectorEnv[
        HeteroBatchData, NDArray[np.int64], NDArray[np.int64]
    ] = (
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
        log_alpha = None
        a_optimizer = None

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_action_space,
        device,
        seed=args.seed,
        n_envs=args.num_envs,
    )
    start_time = time.time()
    returns: deque[float] = deque(maxlen=args.logging_interval)
    lengths: deque[int] = deque(maxlen=args.logging_interval)

    # Since the number of actions per node is constant, we can precompute the target entropy
    target_a = -args.target_entropy_scale_a1 * torch.log(
        1 / torch.tensor(envs.single_action_space.nvec[0])  # type: ignore
    )

    step = step_fn(
        start_time,
        actor,
        q_net,
        log_alpha,
        target_net,
        a_optimizer,
        q_optimizer,
        device,
        target_a,
        actor_optimizer,
        envs,
        pbar,
        args,
        rng=_rng,
    )

    obs: Iterable[HeteroObsData]
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        obs, alpha, returns, lengths = step(
            obs,
            rb,
            global_step,
            returns,
            lengths,
            alpha,
        )

    envs.close()
    return actor


if __name__ == "__main__":
    train(tyro.cli(SACArgs))
