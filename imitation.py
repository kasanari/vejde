from collections import deque
import random
from typing import Any
from kg_gnn import KGGNNAgent
from wrappers.wrapper import GroundedRDDLGraphWrapper
from wrappers.kg_wrapper import register_env
import numpy as np
import torch as th
from torch_geometric.data import Batch, Data
import gymnasium as gym
from gymnasium.spaces import Dict, MultiDiscrete


class MLPAgent(th.nn.Module):
    def __init__(
        self,
        n_types: int,
        n_actions: int,
        layers: int,
        embedding_dim: int,
        activation: th.nn.Module,
    ):
        super().__init__()
        self.embedding = th.nn.Embedding(n_types, embedding_dim)
        self.mlp = th.nn.Sequential(
            th.nn.Linear(n_types * embedding_dim, embedding_dim),
            activation,
            th.nn.Linear(embedding_dim, embedding_dim),
            activation,
        )

        self.nullary_action = th.nn.Linear(embedding_dim, n_actions)
        self.unary_action = th.nn.Linear(embedding_dim, n_types)
        # self.unary_given_nullary_action = th.nn.Linear(
        #     embedding_dim, n_types * n_actions
        # )

    def forward(
        self, a: th.Tensor, x: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        e = self.embedding(x)
        e = e.view(e.size(0), -1)
        logits = self.mlp(e)

        nullary_action = a[:, 0].unsqueeze(1)
        unary_action = a[:, 1].unsqueeze(1)

        p_nullary = th.nn.functional.softmax(self.nullary_action(logits), dim=-1)
        p_unary = th.nn.functional.softmax(self.unary_action(logits), dim=-1)

        # l_joint = self.unary_given_nullary_action(logits).view(
        #     -1,
        #     a.size(1),
        #     x.size(1),
        # )

        # conditional_logits = l_joint.gather(
        #     2, nullary_action.unsqueeze(-1).expand(-1, -1, x.size(1))
        # ).squeeze(1)

        # p_unary_given_nullary = th.nn.functional.softmax(
        #     self.unary_action(logits).view(-1, x.size(0), a.size(0)), dim=-1
        # )

        p_a_nullary = p_nullary.gather(1, nullary_action)
        p_a_unary = p_unary.gather(1, unary_action)
        logprob = th.log(p_a_nullary * p_a_unary)

        return logprob

    def sample_action(self, x: th.Tensor):
        e = self.embedding(x)
        e = e.view(e.size(0), -1)
        logits = self.mlp(e)

        p_nullary = th.nn.functional.softmax(self.nullary_action(logits), dim=-1)
        p_unary = th.nn.functional.softmax(self.unary_action(logits), dim=-1)

        nullary_action = th.distributions.Categorical(p_nullary).sample()
        unary_action = th.distributions.Categorical(p_unary).sample()

        return th.stack([nullary_action, unary_action], dim=1)


def dict_to_data(obs: dict[str, tuple[Any]]) -> list[Data]:
    return Data(
        x=th.as_tensor(obs["nodes"], dtype=th.int64),
        edge_index=th.as_tensor(obs["edge_index"], dtype=th.int64).T,
        edge_attr=th.as_tensor(obs["edge_attr"], dtype=th.int64),
    )


def test_imitation():
    env_id = register_env()
    domain = "rddl/conditional_bandit.rddl"
    instance = "rddl/conditional_bandit_i0.rddl"

    th.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    env: gym.Env[Dict, MultiDiscrete] = gym.make(
        env_id,
        domain=domain,
        instance=instance,
        add_inverse_relations=False,
        types_instead_of_objects=False,
        render_mode="idx",
    )
    n_types = env.observation_space.spaces["nodes"].feature_space.n
    n_relations = env.observation_space["edge_attr"].feature_space.n
    n_actions = env.action_space.nvec[0]
    # gnn_agent = KGGNNAgent(
    #     n_types,
    #     n_relations,
    #     n_actions,
    #     layers=1,
    #     embedding_dim=2,
    #     activation=th.nn.LeakyReLU(),
    #     aggregation="max",
    # )
    agent = MLPAgent(
        n_types, n_actions, layers=1, embedding_dim=2, activation=th.nn.LeakyReLU()
    )
    optimizer = th.optim.AdamW(agent.parameters(), lr=0.01, amsgrad=True)

    return_ = [evaluate(env, agent, 0) for i in range(10)]
    print(np.mean(return_))

    _ = [iteration(i, env, agent, optimizer, 0) for i in range(1000)]

    return_ = [evaluate(env, agent, 0) for i in range(10)]
    print(np.mean(return_))

    th.save(agent.state_dict(), "mlp_agent.pth")


def iteration(i, env, agent, optimizer, seed: int):
    obs, actions = rollout(env, seed)
    obs = [dict_to_data(o) for o in obs]
    loss, grad_norm = update(i, agent, optimizer, actions, obs)
    print(f"{i} Loss: {loss:.3f}, Grad Norm: {grad_norm:.3f}")
    return loss


def update(
    iteration: int,
    agent: th.nn.Module,
    optimizer: th.optim.Optimizer,
    actions,
    obs: list[Data],
):
    l2_weight = 0.0
    # b = Batch.from_data_list(obs)
    b = th.stack([d.x for d in obs])
    logprob = agent.forward(
        th.as_tensor(actions),
        b,
        # b.edge_index,
        # b.edge_attr,
        # b.batch,
    )

    l2_norms = [th.sum(th.square(w)) for w in agent.parameters()]
    l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square
    l2_loss = l2_weight * l2_norm
    loss = -logprob.mean() + l2_loss

    optimizer.zero_grad()
    loss.backward()

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
        time += 1
        # action = env.action_space.sample()
        action = [1, 3]
        obs, reward, terminated, truncated, info = env.step(action)

        # env.render()
        # exit()
        done = terminated or truncated

        sum_reward += reward
        actions_buf.append(action)
        obs_buf.append(obs)

        # print(obs)
        # print(action)
        # print(reward)

    # print(f"Episode {seed}: {sum_reward}, {avg_loss}, {avg_l2_loss}")

    return list(obs_buf), list(actions_buf)


def evaluate(env: gym.Env, agent: th.nn.Module, seed: int):
    obs, info = env.reset(seed=seed)
    done = False
    time = 0
    sum_reward = 0
    while not done:
        time += 1
        # action = env.action_space.sample()

        # b = Batch.from_data_list([dict_to_data(obs)])
        b = th.as_tensor(obs["nodes"], dtype=th.int64).unsqueeze(0)

        action = agent.sample_action(
            b,
            # b.edge_index,
            # b.edge_attr,
            # b.batch,
        )

        obs, reward, terminated, truncated, info = env.step(action.squeeze(0))

        # env.render()
        # exit()
        done = terminated or truncated
        sum_reward += reward
        # print(obs)
        # print(action)
        # print(reward)

    return sum_reward


if __name__ == "__main__":
    test_imitation()
