# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
from typing import Any

from geometric import GNNAgent
from wrappers.kg_wrapper import KGRDDLGraphWrapper

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import tensordict
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import tyro
import wandb
from tensordict import from_module
from tensordict.nn import CudaGraphModule
from torch import Tensor
from torch_geometric.data import Batch, Data


def dict_to_data(data: dict[str, tuple[Any]], num_envs: int) -> list[Data]:
    return [
        Data(
            x=torch.as_tensor(data["nodes"][n], dtype=torch.int64),
            edge_index=torch.as_tensor(data["edge_index"][n], dtype=torch.int64).T,
            edge_attr=torch.as_tensor(data["edge_attr"][n], dtype=torch.int64),
        )
        for n in range(num_envs)
    ]


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
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-5
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 64
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.001
    """coefficient of the value function"""
    max_grad_norm: float = 100
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    measure_burnin: int = 3
    """Number of burn-in iterations for speed measure."""

    compile: bool = False
    """whether to use torch.compile."""
    cudagraphs: bool = False
    """whether to use cudagraphs on top of compile."""


def make_env(domain: str, instance: int, idx: int, capture_video: bool, run_name: str):
    def thunk():
        if capture_video and idx == 0:
            env = KGRDDLGraphWrapper(
                env_id,
                render_mode="rgb_array",
            )
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = KGRDDLGraphWrapper(domain, instance)
        # env = gym.wrappers.FlattenObservation(
        #     env
        # )  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(
        self, n_types: int, n_relations: int, n_actions: int, device: str | None = None
    ):
        super().__init__()
        self.gnn_agent = GNNAgent(n_types, n_relations, n_actions)

    def get_value(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch_idx: Tensor,
    ):
        _, _, _, value = self.gnn_agent(x, edge_index, edge_attr, batch_idx)
        return value

    def get_action_and_value(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch_idx: Tensor,
        action: Tensor | None = None,
    ):
        action, logprob, entropy, value = self.gnn_agent(
            x, edge_index, edge_attr, batch_idx
        )
        return (
            action,
            logprob,
            entropy,
            value,
        )


def gae(next_obs, next_done, container: tensordict.TensorDict) -> tensordict.TensorDict:
    # bootstrap value if not done
    b = Batch.from_data_list(next_obs)
    next_value = get_value(b.x, b.edge_index, b.edge_attr, b.batch).reshape(-1)
    lastgaelam = 0
    nextnonterminals = (~container["dones"]).float().unbind(0)
    vals = container["vals"]
    vals_unbind = vals.unbind(0)
    rewards = container["rewards"].unbind(0)

    advantages = []
    nextnonterminal = (~next_done).float()
    nextvalues = next_value
    for t in range(args.num_steps - 1, -1, -1):
        cur_val = vals_unbind[t]
        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - cur_val
        advantages.append(
            delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        )
        lastgaelam = advantages[-1]

        nextnonterminal = nextnonterminals[t]
        nextvalues = cur_val

    advantages = container["advantages"] = torch.stack(list(reversed(advantages)))
    container["returns"] = advantages + vals
    return container


def rollout(obs: list[Data], done: bool):
    ts: list[tensordict.TensorDict] = []
    data = []
    returns = deque()
    lengths = deque()
    for _ in range(args.num_steps):
        # ALGO LOGIC: action logic
        batch = Batch.from_data_list(obs)
        action, logprob, _, value = policy(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, next_done, infos = step_func(action)

        if "final_info" in infos:
            for info in infos["final_info"]:
                r = float(info["episode"]["r"].reshape(()))
                l = int(info["episode"]["l"].reshape(()))
                # max_ep_ret = max(max_ep_ret, r)
                returns.append(r)
                lengths.append(l)
            # desc = f"global_step={global_step}, episodic_return={torch.tensor(avg_returns).mean(): 4.2f} (max={max_ep_ret: 4.2f})"

        ts.append(
            tensordict.TensorDict._new_unsafe(
                # obs=obs,
                # cleanrl ppo examples associate the done with the previous obs (not the done resulting from action)
                dones=done,
                vals=value.flatten(),
                actions=action,
                logprobs=logprob,
                rewards=reward,
                batch_size=(args.num_envs,),
            )
        )

        data.extend(obs)
        next_obs = dict_to_data(next_obs, args.num_envs)
        obs = next_obs  # = next_obs.to(device, non_blocking=True)
        done = next_done.to(device, non_blocking=True)

    container = torch.stack(ts, 0).to(device)
    batch = Batch.from_data_list(data).to(device)
    return next_obs, done, container, batch, returns, lengths


def update(
    x: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor,
    batch_idx: Tensor,
    actions: Tensor,
    logprobs: Tensor,
    advantages: Tensor,
    returns: Tensor,
    vals: Tensor,
):
    optimizer.zero_grad()
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
        x, edge_index, edge_attr, batch_idx, actions
    )
    logratio = newlogprob - logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = vals + torch.clamp(
            newvalue - vals,
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    loss.backward()
    gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()

    return {
        "approx_kl": approx_kl,
        "v_loss": v_loss.detach(),
        "pg_loss": pg_loss.detach(),
        "entropy_loss": entropy_loss.detach(),
        "old_approx_kl": old_approx_kl,
        "clipfrac": clipfrac,
        "gn": gn,
    }


# TODO maybe add this back
# update = TensorDictModule(
#     update,
#     in_keys=[
#         "x",
#         "edge_index",
#         "edge_attr",
#         "batch_idx",
#         "actions",
#         "logprobs",
#         "advantages",
#         "returns",
#         "vals",
#     ],
#     out_keys=[
#         "approx_kl",
#         "v_loss",
#         "pg_loss",
#         "entropy_loss",
#         "old_approx_kl",
#         "clipfrac",
#         "gn",
#     ],
# )

if __name__ == "__main__":
    args = tyro.cli(Args)

    batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = batch_size // args.num_minibatches
    args.batch_size = args.num_minibatches * args.minibatch_size
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.compile}__{args.cudagraphs}"

    domain = "Elevators_MDP_ippc2011"
    instance = 1

    wandb.init(
        project="ppo_rddl_gnn",
        name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
        config=vars(args),
        save_code=True,
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ####### Environment setup #######
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(domain, instance, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ]
    )

    n_types = envs.single_observation_space.spaces["nodes"].feature_space.n
    n_relations = envs.single_observation_space.spaces["edge_attr"].feature_space.n
    n_actions = envs.single_action_space.nvec[1]

    # Register step as a special op not to graph break
    # @torch.library.custom_op("mylib::step", mutates_args=())
    def step_func(
        action: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        next_obs_np, reward, terminations, truncations, info = envs.step(
            action.cpu().numpy()
        )
        next_done = np.logical_or(terminations, truncations)
        return (
            next_obs_np,
            torch.as_tensor(reward),
            torch.as_tensor(next_done),
            info,
        )

    ####### Agent #######
    agent = Agent(n_types, n_relations, n_actions, device=device)
    # Make a version of agent with detached params
    agent_inference = Agent(n_types, n_relations, n_actions, device=device)
    agent_inference_p = from_module(agent).data
    agent_inference_p.to_module(agent_inference)

    ####### Optimizer #######
    optimizer = optim.Adam(
        agent.parameters(),
        lr=torch.tensor(args.learning_rate, device=device),
        eps=1e-5,
        capturable=args.cudagraphs and not args.compile,
    )

    ####### Executables #######
    # Define networks: wrapping the policy in a TensorDictModule allows us to use CudaGraphModule
    policy = agent_inference.get_action_and_value
    get_value = agent_inference.get_value

    # Compile policy
    if args.compile:
        policy = torch.compile(policy)
        gae = torch.compile(gae, fullgraph=True)
        update = torch.compile(update)

    if args.cudagraphs:
        policy = CudaGraphModule(policy)
        gae = CudaGraphModule(gae)
        update = CudaGraphModule(update)

    avg_returns = deque(maxlen=20)
    avg_lengths = deque(maxlen=20)
    global_step = 0
    container_local = None
    next_obs = envs.reset()[0]
    next_obs = dict_to_data(next_obs, args.num_envs)
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)
    # max_ep_ret = -float("inf")
    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))
    # desc = ""
    global_step_burnin = None
    for iteration in pbar:
        if iteration == args.measure_burnin:
            global_step_burnin = global_step
            start_time = time.time()

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        torch.compiler.cudagraph_mark_step_begin()
        next_obs, next_done, container, batch, returns, lengths = rollout(
            next_obs, next_done
        )
        avg_returns.extend(returns)
        avg_lengths.extend(lengths)

        global_step += container.numel()

        container = gae(next_obs, next_done, container)
        container_flat = container.view(-1)

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(container_flat.shape[0], device=device).split(
                args.minibatch_size
            )
            for b in b_inds:
                container_local = container_flat[b]
                minibatch = Batch.from_data_list(batch.index_select(b))
                # container_local["x"] = minibatch.x
                # container_local["edge_index"] = minibatch.edge_index
                # container_local["edge_attr"] = minibatch.edge_attr
                # container_local["batch_idx"] = minibatch.batch

                # out = update(container_local, tensordict_out=tensordict.TensorDict())
                out = update(
                    minibatch.x,
                    minibatch.edge_index,
                    minibatch.edge_attr,
                    minibatch.batch,
                    container_local["actions"],
                    container_local["logprobs"],
                    container_local["advantages"],
                    container_local["returns"],
                    container_local["vals"],
                )
                if args.target_kl is not None and out["approx_kl"] > args.target_kl:
                    break
            else:
                continue
            break

        if global_step_burnin is not None and iteration % 1 == 0:
            speed = (global_step - global_step_burnin) / (time.time() - start_time)
            r = container["rewards"].mean()
            r_max = container["rewards"].max()
            avg_returns_t = torch.tensor(avg_returns).mean()
            avg_lengths_t = torch.tensor(avg_lengths).mean()

            with torch.no_grad():
                logs = {
                    "episode_return": np.array(avg_returns).mean(),
                    "episode_length": np.array(avg_lengths).mean(),
                    "logprobs": container["logprobs"].mean(),
                    "advantages": container["advantages"].mean(),
                    "returns": container["returns"].mean(),
                    "vals": container["vals"].mean(),
                    "avg_gradient_norm": out["gn"].mean(),
                    "v_loss": out["v_loss"].mean(),
                    "pg_loss": out["pg_loss"].mean(),
                    "entropy_loss": out["entropy_loss"].mean(),
                    "approx_kl": out["approx_kl"].mean(),
                    "clipfrac": out["clipfrac"].mean(),
                }

            lr = optimizer.param_groups[0]["lr"]
            pbar.set_description(
                f"speed: {speed: 4.1f} sps, "
                f"reward avg: {r :4.2f}, "
                f"reward max: {r_max:4.2f}, "
                f"returns: {avg_returns_t: 4.2f},"
                f"length: {avg_lengths_t: 4.2f},"
                f"lr: {lr: 4.2f}"
            )
            wandb.log(
                {
                    "speed": speed,
                    "episode_return": avg_returns_t,
                    "r_mean": r,
                    "r_max": r_max,
                    "lr": lr,
                    **logs,
                },
                step=global_step,
            )

    envs.close()
