from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
import torch
from gnn_policy.functional import sample_node_then_action
from kg_wrapper import KGRDDLGraphWrapper
from torch import Tensor, LongTensor
import torch.nn as nn
import random
import numpy as np
from numpy.typing import NDArray

from torch_geometric.nn import AttentionalAggregation


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class BipartiteData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return torch.tensor([[self.x_p.size(0)], [self.x_o.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)


class GNNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, num_relations: int):
        super(GNNConv, self).__init__(aggr="max")
        self.root = layer_init(nn.Linear(in_channels, out_channels))
        self.combine = layer_init(nn.Linear(out_channels * 2, out_channels))
        self.relation_embedding = nn.Embedding(num_relations, in_channels)
        nn.init.orthogonal_(self.relation_embedding.weight)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        return self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
        )

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return self.relation_embedding(edge_attr) * x_j  # + edge_attr

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        return x + self.combine(torch.concatenate([x, aggr_out], dim=-1))


class GNN(nn.Module):
    def __init__(self, num_object_classes: int, num_predicate_classes: int):
        super().__init__()
        self.obj_embedding = nn.Embedding(num_object_classes, 16)
        self.conv1 = GNNConv(16, 16, num_predicate_classes)
        self.conv2 = GNNConv(16, 16, num_predicate_classes)
        self.aggr = AttentionalAggregation(nn.Linear(16, 1), nn.Linear(16, 16))

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch_idx: Tensor
    ):
        x = self.obj_embedding(x)

        h = self.conv1(x, edge_index, edge_attr)
        h = self.conv2(h, edge_index, edge_attr)

        g = self.aggr(
            h, batch_idx
        )  # torch.sum(torch.softmax(self.gate(h), dim=0) * self.attn(h), dim=0)

        return h, g


class GNNPolicy(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        self.node_prob = nn.Linear(16, 1)
        self.action_prob = nn.Linear(16, num_actions)
        self.num_actions = num_actions

    def forward(self, h: Tensor, batch_idx: Tensor):
        node_logits = self.node_prob(h).squeeze()
        action_logits = self.action_prob(h)
        node_mask = torch.ones(node_logits.shape[0], dtype=torch.bool)
        num_graphs = batch_idx.max().item() + 1
        action_mask = torch.ones((num_graphs, self.num_actions), dtype=torch.bool)
        actions, logprob, entropy = sample_node_then_action(
            node_logits, action_logits, node_mask, action_mask, batch_idx
        )
        return actions, logprob, entropy


class GNNAgent(nn.Module):
    def __init__(
        self, num_object_classes: int, num_predicate_classes: int, num_actions: int
    ):
        super().__init__()
        self.gnn = GNN(num_object_classes, num_predicate_classes)
        self.policy = GNNPolicy(num_actions)
        self.vf = nn.Linear(16, 1)

    def value(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch_idx: Batch,
        num_graphs: int,
    ):
        _, g = self.gnn(x, edge_index, edge_attr, batch_idx)
        value = self.vf(g)
        return value

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch_idx: Batch,
    ):
        h, g = self.gnn(x, edge_index, edge_attr, batch_idx)
        actions, logprob, entropy = self.policy(h, batch_idx)
        value = self.vf(g)
        return actions, logprob, entropy, value


def obs_to_data(obs: dict[str, NDArray[np.int64]]) -> Data:
    return Data(
        x=LongTensor(obs["node_classes"]),
        edge_index=LongTensor(obs["edge_index"]).T,
        edge_attr=LongTensor(obs["edge_attr"]),
    )


def main():
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    instance = 1
    domain = "Elevators_MDP_ippc2011"
    env = KGRDDLGraphWrapper(domain, instance)
    gnn = GNNAgent(env.num_types, env.num_relations, len(env.action_values))
    optimizer = torch.optim.Adam(gnn.policy.parameters(), lr=0.01)
    action_space = env.action_space
    action_space.seed(0)
    iterations = 100

    for j in range(iterations):
        obs, info = env.reset(seed=0)
        data = obs_to_data(obs)
        step = 0
        obs_buf = []
        actions = []
        rewards = []
        while True:
            step += 1
            action = {"close-door___e0": 0}

            data = obs_to_data(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            obs_buf.append(data)
            actions.append(action)
            rewards.append(reward)

            if done:
                break

            obs = next_obs

        batch = Batch.from_data_list(obs_buf)
        values = gnn.value(batch)
        returns = torch.zeros_like(values)

        for i in reversed(range(len(rewards))):
            returns[i] = sum(rewards[i:])

        advantages = returns - values

        value_loss = advantages.pow(2).mean()
        print(f"{j}: {value_loss.item()}")
        optimizer.zero_grad()
        value_loss.backward()
        optimizer.step()

    value_loss = advantages.round()

    np.set_printoptions(precision=2)
    print(value_loss.detach().numpy())

    pass


if __name__ == "__main__":
    main()
