import json
import random
from collections import deque
from typing import Any, NamedTuple

import flax.nnx as nn
import gymnasium as gym
import jax
import numpy as np
from gymnasium.spaces import Dict, MultiDiscrete
from jax import Array

Tensor = Array, vmap
from jax.nn import mish
from jax.numpy import array, int32, mean, square
from optax import amsgrad
from util import save_eval_data

from gnn import ActionMode, Config, RecurrentGraphAgent
from gnn.data import (StackedStateData, stacked_dict_to_data,
                      stackedstatedata_from_obs,
                      stackedstatedata_from_single_obs)
from rddl import register_pomdp_env as register_env

# from wrappers.kg_wrapper import register_env


class Serializer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Tensor):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class Rollout(NamedTuple):
    rewards: list[float]
    obs: list[dict[str, Any]]
    actions: list[tuple[int, int]]


def save_rollout(rollout: Rollout, path: str):
    with open(path, "w") as f:
        json.dump(rollout._asdict(), f, cls=Serializer)


def load_rollout(path: str) -> Rollout:
    with open(path, "r") as f:
        data = json.load(f)
    return Rollout(**data)


def compare_rollouts(r1: Rollout, r2: Rollout):
    assert r1.rewards == r2.rewards
    assert r1.actions == r2.actions
    for o1, o2 in zip(r1.obs, r2.obs):
        assert o1.keys() == o2.keys()
        for k in o1:
            if isinstance(o1[k], np.ndarray):
                o = o1[k].tolist()
            else:
                o = o1[k]

            assert o == o2[k], f"{o} != {o2[k]} for {k}"


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

    np.random.seed(0)
    random.seed(0)

    env_id = register_env()
    env: gym.Env[Dict, MultiDiscrete] = gym.make(  # type: ignore
        env_id, domain=domain, instance=instance
    )
    n_types = int(env.observation_space["factor"].feature_space.n)  # type: ignore
    n_relations = int(env.observation_space["var_type"].feature_space.n)  # type: ignore
    n_actions = int(env.action_space.nvec[0])  # type: ignore

    config = Config(
        n_types,
        n_relations,
        n_actions,
        seed=0,
        layers=3,
        embedding_dim=8,
        activation=mish,
        aggregation="sum",
        action_mode=ActionMode.ACTION_THEN_NODE,
    )

    rngs = nn.Rngs(params=0, carry=0)  # rng init

    agent = RecurrentGraphAgent(config, rngs)

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
    optimizer = nn.Optimizer(agent, amsgrad(learning_rate=0.01))

    eval = evaluate(env)
    # data = [eval(agent, 0) for i in range(10)]
    # rewards, _, _ = zip(*data)
    # print(np.mean(np.sum(rewards, axis=1)))

    _ = [iteration(i, env, agent, optimizer, 0) for i in range(500)]

    pass

    data = [eval(agent, 0) for i in range(3)]

    save_eval_data(data)

    # agent.save_agent("blink_enough_bandit.pth")


def iteration(i, env, agent, optimizer, seed: int):
    r, length = rollout(env, seed)

    # save_rollout(r, f"rollouts/rollout_{i}.json")
    # saved_r = load_rollout(f"rollouts/rollout_{i}.json")
    # compare_rollouts(r, saved_r)
    loss, grad_norm = update(i, optimizer, r)
    print(f"{i} Loss: {loss:.3f}, Grad Norm: {grad_norm:.3f}, Length: {length}")
    return loss


def per_sample_grad(agent: nn.Module, b: Tensor, actions: Tensor):
    def compute_loss(params, buffers, batch: Tensor, actions: Tensor):
        l2_weight = 0.0

        logprob = functional_call(
            agent, (params, buffers), (actions.unsqueeze(0), batch.unsqueeze(0))
        )

        l2_norms = [sum(square(w)) for w in agent.parameters()]
        l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square
        l2_loss = l2_weight * l2_norm
        loss = -logprob + l2_loss
        return loss.squeeze()

    params = {k: v.detach() for k, v in agent.named_parameters()}
    buffers = {k: v.detach() for k, v in agent.named_buffers()}

    ft_compute_grad = grad(compute_loss)

    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

    ft_per_sample_grads = ft_compute_sample_grad(
        params, buffers, b, th.as_tensor(actions)
    )

    return ft_per_sample_grads


@nn.jit
def update(
    iteration: int,
    optimizer: nn.Optimizer,
    rollout: Rollout,
) -> tuple[float, float]:
    _, obs, actions = rollout
    obs = [stacked_dict_to_data(o) for o in obs]
    s = stackedstatedata_from_obs(obs)
    # b = th.stack([d.var_value for d in obs])
    actions = array(actions, dtype=int32)
    loss, grad = nn.value_and_grad(loss_fn)(optimizer.model, actions, s)

    # d = dict(agent.named_parameters())
    # grads = {k: th.nonzero((-d[k].grad).clamp(min=0)) for k in d}
    # grads = {k: v for k, v in grads.items() if len(v) > 0}

    # if iteration % 100 == 0:
    #     per_sample_grads = per_sample_grad(agent, b, actions)

    # grad_norms = {
    #     k: round(grad_norm_(param, 1.0).item(), 3)
    #     for k, param in agent.named_parameters()
    # }
    # max_grad_norm = max(grad_norms.values())

    # grad_norm = th.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)

    optimizer.update(grad)

    return loss, 0.0


def loss_fn(agent: RecurrentGraphAgent, actions: Tensor, s: StackedStateData) -> Tensor:
    # l2_weight = 0.0
    # l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square
    # l2_loss = l2_weight * l2_norm
    logprob, _, _ = agent(
        actions,
        s,
    )
    loss = -mean(logprob)  # + l2_loss
    return loss


def rollout(env: gym.Env[Dict, MultiDiscrete], seed: int) -> Rollout:
    obs, info = env.reset(seed=seed)
    done = False
    time = 0
    sum_reward: float = 0.0
    actions_buf: deque[tuple[int, int]] = deque()
    obs_buf: deque[dict[str, Any]] = deque()
    done_buf: deque[bool] = deque()
    states = {k: [v] for k, v in info["rddl_state"].items()}
    rewards_buf: deque[float] = deque()
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
        rewards_buf.append(float(reward))

        obs = next_obs
        time += 1

        # print(obs)
        # print(action)
        # print(reward)

    # print(f"Episode {seed}: {sum_reward}, {avg_loss}, {avg_l2_loss}")
    if seed == 0:
        assert sum_reward == 2.0, f"Expert policy failed: {sum_reward}"

    return Rollout(list(rewards_buf), list(obs_buf), list(actions_buf)), time


@nn.jit
def sample(agent: RecurrentGraphAgent, obs: dict, key: Tensor) -> Tensor:
    s = stackedstatedata_from_single_obs(obs)

    (
        action,
        *_,
    ) = agent.sample(
        key,
        s,
        1,
        deterministic=False,
    )
    action = action.squeeze(0)
    return action


def evaluate(env: gym.Env):
    def _evaluate(agent: RecurrentGraphAgent, seed: int):
        obs, info = env.reset(seed=seed)
        done = False
        time = 0

        rewards: deque[float] = deque()
        actions: deque[tuple[int, int]] = deque()
        obs_buf: deque[dict[str, Any]] = deque()
        key = jax.random.PRNGKey(seed)

        while not done:
            time += 1
            # action = env.action_space.sample()

            obs_buf.append(info["rddl_state"])
            key, subkey = jax.random.split(key)
            action = sample(agent, obs, subkey)

            next_obs, reward, terminated, truncated, info = env.step(action.tolist())

            # env.render()
            # exit()
            done = terminated or truncated
            actions.append(info["rddl_action"])  # type: ignore
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

    return _evaluate


if __name__ == "__main__":
    import jax.profiler

    # with jax.profiler.trace("jax-trace", create_perfetto_link=True):
    test_imitation()
