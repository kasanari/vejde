import json
import random
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.spaces import Dict, MultiDiscrete
from torch.func import functional_call, grad, vmap
from torch_geometric.data import Batch, Data
from gnn.data import statedata_from_single_obs, dict_to_data, statedata_from_obs

from gnn import ActionMode, Config, GraphAgent, StateData

# from wrappers.kg_wrapper import register_env
from test.util import save_eval_data
from wrappers.wrapper import register_env


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
    if state["light___r_m"]:
        return [1, 3]

    if state["light___g_m"]:
        return [1, 1]

    return [0, 0]


def test_imitation():
    domain = "rddl/conditional_bandit.rddl"
    instance = "rddl/conditional_bandit_i0.rddl"

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
        render_mode="idx",
    )
    n_objects = env.observation_space.spaces["factor"].shape[0]
    n_vars = env.observation_space["var_value"].shape[0]
    n_types = int(env.observation_space["factor"].high[0])
    n_relations = int(env.observation_space["var_type"].high[0])
    n_actions = int(env.action_space.nvec[0])

    config = Config(
        n_types,
        n_relations,
        n_actions,
        layers=4,
        embedding_dim=4,
        activation=th.nn.LeakyReLU(),
        aggregation="sum",
        action_mode=ActionMode.ACTION_THEN_NODE,
    )

    agent = GraphAgent(
        config,
    )

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

    data = [evaluate(env, agent, 0) for i in range(10)]
    rewards, _, _ = zip(*data)
    print(np.mean(np.sum(rewards, axis=1)))

    _ = [iteration(i, env, agent, optimizer, 0) for i in range(100)]

    pass

    data = [evaluate(env, agent, 0) for i in range(3)]

    save_eval_data(data)

    agent.save_agent("conditional_bandit.pth")


def iteration(i, env, agent, optimizer, seed: int):
    obs, actions = rollout(env, seed)
    obs = [dict_to_data(o) for o in obs]
    loss, grad_norm = update(i, agent, optimizer, actions, obs)
    print(f"{i} Loss: {loss:.3f}, Grad Norm: {grad_norm:.3f}")
    return loss


def per_sample_grad(agent: th.nn.Module, b: th.Tensor, actions: th.Tensor):
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
        params, buffers, b, th.as_tensor(actions)
    )

    return ft_per_sample_grads


def update(
    iteration: int,
    agent: GraphAgent,
    optimizer: th.optim.Optimizer,
    actions,
    obs: list[Data],
):
    l2_weight = 0.0
    s = statedata_from_obs(obs)
    # b = th.stack([d.var_value for d in obs])
    actions = th.as_tensor(actions, dtype=th.int64)
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

    grad_norm = th.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)

    optimizer.step()

    return loss.item(), grad_norm.item()


def rollout(env: gym.Env, seed: int):
    obs, info = env.reset(seed=seed)
    done = False
    time = 0
    sum_reward = 0
    actions_buf = deque()
    obs_buf = deque()
    while not done:
        # action = env.action_space.sample()
        action = policy(info["rddl_state"])
        next_obs, reward, terminated, truncated, info = env.step(action)

        # env.render()
        # exit()
        done = terminated or truncated

        sum_reward += reward
        actions_buf.append(action)
        obs_buf.append(obs)

        obs = next_obs
        time += 1

        # print(obs)
        # print(action)
        # print(reward)

        # print(f"Episode {seed}: {sum_reward}, {avg_loss}, {avg_l2_loss}")

    assert sum_reward == 4.0, f"Expert policy failed: {sum_reward}"

    return list(obs_buf), list(actions_buf)


@th.no_grad()
def evaluate(env: gym.Env, agent: GraphAgent, seed: int):
    obs, info = env.reset(seed=seed)
    done = False
    time = 0

    rewards = deque()
    obs_buf = deque()
    actions = deque()

    while not done:
        time += 1
        # action = env.action_space.sample()

        obs_buf.append(env.unwrapped.last_rddl_obs)

        s = statedata_from_single_obs(obs)
        # b = {k: th.as_tensor(v, dtype=th.float32) for k, v in obs["nodes"].items()}
        # b = th.as_tensor(obs["var_value"], dtype=th.float32)

        action, _, _, _ = agent.sample(
            s,
            deterministic=True,
        )

        next_obs, reward, terminated, truncated, info = env.step(action.squeeze(0))

        # env.render()
        # exit()
        done = terminated or truncated
        actions.append(env.unwrapped.last_action_values)
        rewards.append(reward)
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
