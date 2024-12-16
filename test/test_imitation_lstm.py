import json
import random
from collections import deque
from typing import Any, Iterable

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.spaces import Dict, MultiDiscrete
from torch.func import functional_call, grad, vmap
from torch_geometric.data import Data

from torch.utils._foreach_utils import (
    _group_tensors_by_device_and_dtype,
)

from gnn import ActionMode, Config, RecurrentGraphAgent

# from wrappers.kg_wrapper import register_env
from gnn.data import (
    stacked_dict_to_data,
    stackedstatedata_from_obs,
    stackedstatedata_from_single_obs,
)
from util import save_eval_data
from rddl import register_pomdp_env as register_env


class Serializer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, th.Tensor):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return super().default(obj)


@th.no_grad()  # type: ignore
def grad_norm_(
    parameters: th.Tensor | Iterable[th.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> th.Tensor:
    if isinstance(parameters, th.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return th.tensor(0.0)
    first_device = grads[0].device
    grouped_grads: dict[
        tuple[th.device, th.dtype], tuple[list[list[th.Tensor]], list[int]]
    ] = _group_tensors_by_device_and_dtype([grads])  # type: ignore[assignment]

    norms: list[th.Tensor] = []
    for (device, _), ([device_grads], _) in grouped_grads.items():  # type: ignore[assignment]
        norms.extend([th.linalg.vector_norm(g, norm_type) for g in device_grads])

    total_norm = th.linalg.vector_norm(
        th.stack([norm.to(first_device) for norm in norms]), norm_type
    )

    if error_if_nonfinite and th.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )

    return total_norm


def knowledge_graph_policy(obs):
    nonzero = np.flatnonzero(obs["edge_attr"])
    if len(nonzero) == 0:
        return [0, 0]

    obj = next(iter(nonzero))

    obj = obs["edge_index"][obj][0]

    button = next(
        p for p, c in obs["edge_index"] if c == obj and not ((p, c) == (obj, obj))
    )

    return [1, button]


def policy(state):
    if (
        state["enough_light___r_m"]
        and state["light___r_m"]
        and not state["empty___r_m"]
    ):
        return (1, 4)

    if (
        state["enough_light___g_m"]
        and state["light___g_m"]
        and not state["empty___g_m"]
    ):
        return (1, 2)

    return (0, 0)


def counting_policy(state):
    if np.array(state["light_observed___r_m"], dtype=bool).sum() == 3:
        return [1, 3]

    if np.array(state["light_observed___g_m"], dtype=bool).sum() == 3:
        return [1, 1]

    return [0, 0]


def count_above_policy(state):
    if np.array(state["light_observed___r_m"], dtype=bool).sum() > 3:
        return [1, 3]

    if np.array(state["light_observed___g_m"], dtype=bool).sum() > 3:
        return [1, 1]

    return [0, 0]


# def dict_to_data(obs: dict[str, tuple[Any]]) -> list[Data]:
#     return Data(
#         x=th.as_tensor(obs["nodes"], dtype=th.int64),
#         edge_index=th.as_tensor(obs["edge_index"], dtype=th.int64).T,
#         edge_attr=th.as_tensor(obs["edge_attr"], dtype=th.int64),
#     )


def test_imitation():
    domain = "rddl/blink_enough_bandit.rddl"
    instance = "rddl/blink_enough_bandit_i0.rddl"

    th.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    env_id = register_env()
    env: gym.Env[Dict, MultiDiscrete] = gym.make(
        env_id,
        domain=domain,
        instance=instance,
        # add_inverse_relations=False,
        # types_instead_of_objects=False,
    )
    n_types = int(env.observation_space["factor"].feature_space.n)
    n_relations = int(env.observation_space["var_type"].feature_space.n)
    n_actions = int(env.action_space.nvec[0])

    config = Config(
        n_types,
        n_relations,
        n_actions,
        layers=3,
        embedding_dim=8,
        activation=th.nn.Mish(),
        aggregation="sum",
        action_mode=ActionMode.ACTION_THEN_NODE,
    )

    agent = RecurrentGraphAgent(config)

    # state_dict = th.load("bipart_gnn_agent.pth", weights_only=True)
    # agent.load_state_dict(state_dict)

    # agent = MLPAgent(
    #     n_types,
    #     n_relations,
    #     n_actions,
    #     layers=1,
    #     embedding_dim=4,
    #     activation=th.nn.ReLU(),
    # )
    optimizer = th.optim.AdamW(agent.parameters(), lr=0.01, amsgrad=True)

    data = [evaluate(env, agent, i) for i in range(10)]
    rewards, _, _ = zip(*data)
    print(np.mean([np.sum(r) for r in rewards]))

    num_seeds = 10

    _ = [iteration(i, env, agent, optimizer, i % num_seeds) for i in range(100)]

    pass

    data = [evaluate(env, agent, i) for i in range(3)]

    save_eval_data(data)

    agent.save_agent("blink_enough_bandit.pth")


def iteration(i, env, agent, optimizer, seed: int):
    obs, actions, length = rollout(env, seed)
    obs = [stacked_dict_to_data(o) for o in obs]
    loss, grad_norm = update(i, agent, optimizer, actions, obs)
    print(f"{i} Loss: {loss:.3f}, Grad Norm: {grad_norm:.3f}, Length: {length}")
    return loss


def per_sample_grad(agent: th.nn.Module, s, actions: th.Tensor):
    def compute_loss(params, buffers, batch: th.Tensor, actions: th.Tensor):
        l2_weight = 0.0

        logprob = functional_call(
            agent, (params, buffers), (actions.unsqueeze(0), batch.unsqueeze(0))
        )

        l2_norms = [th.sum(th.square(w)) for w in agent.parameters()]
        l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square
        l2_loss = l2_weight * l2_norm
        loss = -logprob + l2_loss
        return loss.squeeze()

    params = {k: v.detach() for k, v in agent.named_parameters()}
    buffers = {k: v.detach() for k, v in agent.named_buffers()}

    ft_compute_grad = grad(compute_loss)

    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

    ft_per_sample_grads = ft_compute_sample_grad(
        params, buffers, s, th.as_tensor(actions)
    )

    return ft_per_sample_grads


def update(
    iteration: int,
    agent: th.nn.Module,
    optimizer: th.optim.Optimizer,
    actions: th.Tensor,
    obs: list[Data],
):
    l2_weight = 0.0
    s = stackedstatedata_from_obs(obs)
    # b = th.stack([d.var_value for d in obs])
    logprob, _, _ = agent.forward(
        actions,
        s,
    )

    l2_norms = [th.sum(th.square(w)) for w in agent.parameters()]
    l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square
    l2_loss = l2_weight * l2_norm
    loss = -logprob.mean() + l2_loss

    optimizer.zero_grad()
    loss.backward()

    d = dict(agent.named_parameters())
    grads = {k: th.nonzero((-d[k].grad).clamp(min=0)) for k in d}
    grads = {k: v for k, v in grads.items() if len(v) > 0}

    # if iteration % 100 == 0:
    #     per_sample_grads = per_sample_grad(agent, b, actions)

    grad_norms = {
        k: round(grad_norm_(param, 1.0).item(), 3)
        for k, param in agent.named_parameters()
    }
    max_grad_norm = max(grad_norms.values())

    grad_norm = th.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)

    optimizer.step()

    return loss.item(), grad_norm


def rollout(env: gym.Env[Dict, MultiDiscrete], seed: int):
    obs, info = env.reset(seed=seed)
    done = False
    time = 0
    sum_reward: float = 0.0
    actions_buf: deque[tuple[int, int]] = deque()
    obs_buf: deque[dict[str, Any]] = deque()
    done_buf: deque[bool] = deque()
    states = {k: [v] for k, v in info["rddl_state"].items()}
    while not done:
        # action = env.action_space.sample()
        action = policy(info["rddl_state"])
        next_obs, reward, terminated, truncated, info = env.step(action)  # type: ignore

        # env.render()
        # exit()
        done = terminated or truncated
        if not done:
            states = {k: v + [info["rddl_state"][k]] for k, v in states.items()}

        sum_reward += float(reward)
        actions_buf.append(action)
        obs_buf.append(obs)  # type: ignore
        done_buf.append(done)

        obs = next_obs
        time += 1

        # print(obs)
        # print(action)
        # print(reward)

    # print(f"Episode {seed}: {sum_reward}, {avg_loss}, {avg_l2_loss}")

    assert sum_reward == 2.0, f"Expert policy failed: {sum_reward}"

    return list(obs_buf), th.as_tensor(list(actions_buf), dtype=th.int64), time


@th.no_grad()  # type: ignore
def evaluate(env: gym.Env[Dict, MultiDiscrete], agent: RecurrentGraphAgent, seed: int):
    obs, info = env.reset(seed=seed)
    done = False
    time = 0

    rewards: deque[float] = deque()
    actions: deque[tuple[int, int]] = deque()
    obs_buf: deque[dict[str, Any]] = deque()

    while not done:
        time += 1
        # action = env.action_space.sample()

        obs_buf.append(info["rddl_state"])  # type: ignore

        s = stackedstatedata_from_single_obs(obs)  # type: ignore
        # b = {k: th.as_tensor(v, dtype=th.float32) for k, v in obs["nodes"].items()}
        # b = th.as_tensor(obs["var_value"], dtype=th.float32)

        action, _, _, _ = agent.sample(
            s,
            deterministic=True,
        )

        next_obs, reward, terminated, truncated, info = env.step(action.squeeze(0))  # type: ignore

        # env.render()
        # exit()
        done = terminated or truncated
        actions.append(actions.append(info["rddl_action"]))  # type: ignore
        rewards.append(float(reward))
        obs = next_obs
        # print(obs)
        # print(action)
        # print(reward)

    return (
        list(rewards),
        list(obs_buf),
        list(actions),
    )


if __name__ == "__main__":
    test_imitation()
