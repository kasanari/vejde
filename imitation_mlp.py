from collections import deque
import random
from typing import Any

# from wrappers.kg_wrapper import register_env
from wrappers.wrapper import register_env
import numpy as np
import torch as th
from torch_geometric.data import Batch, Data
import gymnasium as gym
from gymnasium.spaces import Dict, MultiDiscrete

from torch.func import functional_call, grad, vmap
import json


class Serializer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, th.Tensor):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def policy(state):
    if state["light___r_m"]:
        return [1, 3]

    if state["light___g_m"]:
        return [1, 1]

    return [0, 0]


class MLPAgent(th.nn.Module):
    def __init__(
        self,
        n_types: int,
        n_relations: int,
        n_actions: int,
        layers: int,
        embedding_dim: int,
        activation: th.nn.Module,
    ):
        super().__init__()
        # self.embedding = th.nn.Embedding(n_types, embedding_dim)
        # th.nn.init.constant_(self.embedding.weight, 1.0)
        self.mlp = th.nn.Sequential(
            th.nn.Linear(n_relations, embedding_dim),
            activation,
            # th.nn.Linear(embedding_dim, embedding_dim),
            # activation,
            # th.nn.Linear(embedding_dim, embedding_dim),
            # activation,
        )

        # self.embedders = th.nn.ModuleDict(
        #     {
        #         "button": th.nn.Linear(1, embedding_dim),
        #         "machine": th.nn.Linear(1, embedding_dim),
        #     }
        # )

        # th.nn.init.constant_(self.mlp[0].weight, 1.0)
        # th.nn.init.constant_(self.mlp[2].weight, 1.0)
        # th.nn.init.constant_(self.mlp[0].bias, 0.0)
        # th.nn.init.constant_(self.mlp[2].bias, 0.0)

        self.nullary_action = th.nn.Linear(embedding_dim, n_actions)
        self.unary_action = th.nn.Linear(embedding_dim, n_types)

        # th.nn.init.constant_(self.nullary_action.weight, 1.0)
        # th.nn.init.constant_(self.unary_action.weight, 1.0)
        # th.nn.init.constant_(self.nullary_action.bias, 0.0)
        # th.nn.init.constant_(self.unary_action.bias, 0.0)

        # self.unary_given_nullary_action = th.nn.Linear(
        #     embedding_dim, n_types * n_actions
        # )

    def forward(
        self, a: th.Tensor, x: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        # e = self.embedding(x)
        # e = e.view(e.size(0), -1)

        # x = x.view(x.size(0), -1)

        logits = self.mlp(x)

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

        # when nullary_action is 0, p_a_unary is 1
        p_a_unary = th.where(nullary_action == 0, th.ones_like(p_a_unary), p_a_unary)

        logprob = th.log(p_a_nullary * p_a_unary)

        return logprob, 0.0, 0.0

    def sample(self, x: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # x = x.view(x.size(0), -1)
        logits = self.mlp(x)

        p_nullary = th.nn.functional.softmax(self.nullary_action(logits), dim=-1)
        p_unary = th.nn.functional.softmax(self.unary_action(logits), dim=-1)

        threshold = 0.05
        # threshold and rescale
        p_unary = th.where(p_unary < threshold, th.zeros_like(p_unary), p_unary)
        p_unary = th.nn.functional.softmax(p_unary, dim=-1)

        nullary_action = (
            th.distributions.Categorical(p_nullary).sample()
            if not deterministic
            else th.argmax(p_nullary)
        )
        unary_action = (
            th.distributions.Categorical(p_unary).sample()
            if not deterministic
            else th.argmax(p_unary)
        )

        return th.stack([nullary_action, unary_action], dim=0), 0.0, 0.0


def dict_to_data(obs: dict[str, tuple[Any]]) -> Data:
    return Data(
        var_value=th.as_tensor(obs["var_value"], dtype=th.float32),
        var_type=th.as_tensor(obs["var_type"], dtype=th.int64),
        factor=th.as_tensor(obs["factor"], dtype=th.int64),
        edge_index=th.as_tensor(obs["edge_index"], dtype=th.int64).T,
        edge_attr=th.as_tensor(obs["edge_attr"], dtype=th.int64),
        num_nodes=obs["factor"].shape[0],  # + obs["var_value"].shape[0]
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
        # add_inverse_relations=False,
        # types_instead_of_objects=False,
        render_mode="idx",
    )
    n_objects = env.observation_space.spaces["factor"].shape[0]
    n_vars = env.observation_space["var_value"].shape[0]
    n_types = env.observation_space["factor"].high[0]
    n_relations = env.observation_space["var_type"].high[0]
    n_actions = env.action_space.nvec[0]

    agent = MLPAgent(
        n_objects,
        n_vars,
        n_actions,
        layers=4,
        embedding_dim=4,
        activation=th.nn.ReLU(),
    )
    optimizer = th.optim.AdamW(agent.parameters(), lr=0.01, amsgrad=True)

    data = [evaluate(env, agent, 0) for i in range(10)]
    rewards, _, _ = zip(*data)
    print(np.mean(np.sum(rewards, axis=1)))

    _ = [iteration(i, env, agent, optimizer, 0) for i in range(100)]

    pass

    data = [evaluate(env, agent, 0) for i in range(3)]

    rewards, _, _ = zip(*data)

    print(np.mean(np.sum(rewards, axis=1)))

    to_write = {
        f"ep_{i}": [
            {
                "reward": r,
                "obs": s,
                "action": a,
            }
            for r, s, a in zip(*episode)
        ]
        for i, episode in enumerate(data)
    }

    with open("evaluation.json", "w") as f:
        import json

        json.dump(to_write, f, cls=Serializer, indent=2)

    th.save(agent.state_dict(), "bipart_gnn_agent.pth")


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
    agent: th.nn.Module,
    optimizer: th.optim.Optimizer,
    actions,
    obs: list[Data],
):
    l2_weight = 0.0
    b = th.stack([d.var_value for d in obs])
    actions = th.as_tensor(actions, dtype=th.int64)
    logprob, _, _ = agent.forward(
        actions,
        b,
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
        time += 1
        # action = env.action_space.sample()
        action = policy(env.unwrapped.last_rddl_obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # env.render()
        # exit()
        done = terminated or truncated

        sum_reward += reward
        actions_buf.append(action)
        obs_buf.append(obs)

        obs = next_obs

        # print(obs)
        # print(action)
        # print(reward)

    # print(f"Episode {seed}: {sum_reward}, {avg_loss}, {avg_l2_loss}")

    if sum_reward != 4.0:
        print("Expert policy failed")

    return list(obs_buf), list(actions_buf)


@th.no_grad()
def evaluate(env: gym.Env, agent: th.nn.Module, seed: int):
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

        # b = Batch.from_data_list([dict_to_data(obs)])
        # b = {k: th.as_tensor(v, dtype=th.float32) for k, v in obs["nodes"].items()}
        b = th.as_tensor(obs["var_value"], dtype=th.float32)

        action, _, _ = agent.sample(
            b,
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
