import random

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import LongTensor, Tensor
from torch.nn.functional import leaky_relu
from torch_geometric.data import Batch, Data
from torch_geometric.nn import AttentionalAggregation, MessagePassing, SumAggregation
from torch_geometric.utils import softmax
from wrappers.kg_wrapper import KGRDDLGraphWrapper

activation_to_str: dict[type[nn.Module], str] = {
    nn.ReLU: "relu",
    nn.LeakyReLU: "leaky_relu",
    nn.Tanh: "tanh",
    nn.Sigmoid: "sigmoid",
    nn.Softmax: "softmax",
}


def layer_init(
    layer: nn.Module,
    nonlinearity: str,
    bias_const: float = 0.0,
):
    # return layer
    gain = torch.nn.init.calculate_gain(nonlinearity, param=None)
    if isinstance(layer, nn.Linear):
        # torch.nn.init.orthogonal_(layer.weight, gain)
        torch.nn.init.constant_(layer.weight, 1.0)
        # torch.nn.init.xavier_normal_(layer.weight, gain)
        torch.nn.init.constant_(layer.bias, 0.0)
    elif isinstance(layer, nn.Embedding):
        # torch.nn.init.xavier_normal_(layer.weight)
        # torch.nn.init.orthogonal_(layer.weight, gain)
        torch.nn.init.constant_(layer.weight, 1.0)
    return layer


class MLPLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: nn.Module):
        super(MLPLayer, self).__init__()
        self.linear = layer_init(
            nn.Linear(in_features, out_features),
            activation_to_str[activation.__class__],
        )
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.linear(x))


class EmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, activation: nn.Module):
        super(EmbeddingLayer, self).__init__()
        self.embedding = layer_init(
            nn.Embedding(num_embeddings, embedding_dim),
            activation_to_str[activation.__class__],
        )
        # torch.nn.init.constant_(self.embedding.weight, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)


class LinearTransform(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: nn.Module):
        super(LinearTransform, self).__init__()
        self.linear = layer_init(
            nn.Linear(in_features, out_features),
            activation_to_str[activation.__class__],
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class GATConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        activation: nn.Module,
        aggregation: str,
    ):
        super(GATConv, self).__init__(aggr=aggregation)

        self.key = nn.Parameter(torch.randn(in_channels, out_channels))
        self.query = nn.Parameter(torch.randn(in_channels, out_channels))
        self.edge_transform = nn.Parameter(torch.randn(in_channels, out_channels))
        self.attn = nn.Parameter(torch.randn(1, out_channels))
        self.residual = nn.Parameter(torch.randn(in_channels, out_channels))
        # self.activation = LeakyReLU()

        # init
        nn.init.xavier_uniform_(self.key)
        nn.init.xavier_uniform_(self.query)
        nn.init.xavier_uniform_(self.edge_transform)
        nn.init.xavier_uniform_(self.attn)
        nn.init.xavier_uniform_(self.residual)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        k = x @ self.key.T
        q = x @ self.query.T
        e = edge_attr @ self.edge_transform.T

        alpha = self.edge_updater(edge_index, x=(k, q), edge_attr=e)

        out = self.propagate(
            edge_index,
            x=x,
            alpha=alpha,
        )

        out = x @ self.residual.T + out

        return out

    def edge_update(
        self,
        x_j: Tensor,
        x_i: Tensor,
        edge_attr: Tensor,
        index: Tensor,
        ptr: Tensor,
        dim_size: Tensor,
    ):
        x = x_j + x_i + edge_attr
        x = leaky_relu(x)
        alpha = (x @ self.attn.T).sum(axis=-1)
        alpha = softmax(alpha, index, ptr, dim_size)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return x_j * alpha.unsqueeze(-1)


class GNNConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        activation: nn.Module,
        aggregation: str = "sum",
    ):
        super(GNNConv, self).__init__(aggr=aggregation)
        self.combine = MLPLayer(in_channels + out_channels, out_channels, activation)
        self.msg_combine = MLPLayer(
            out_channels + out_channels, out_channels, activation
        )
        #

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        return self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
        )

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return self.msg_combine(torch.concatenate([x_j, edge_attr], axis=-1))

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        return x + self.combine(torch.concatenate([x, aggr_out], axis=-1))


class GlobalNode(nn.Module):
    def __init__(self, emb_size: int, activation: nn.Module):
        super().__init__()
        self.aggr = AttentionalAggregation(
            LinearTransform(emb_size, 1, activation),
            LinearTransform(emb_size, emb_size, activation),
        )
        self.linear = MLPLayer(2 * emb_size, emb_size, activation)

    def forward(self, x: Tensor, g_prev: Tensor, batch_idx: Tensor) -> Tensor:
        g = self.aggr(x, batch_idx)
        return g_prev + self.linear(torch.concatenate([g, g_prev], axis=-1))


class SumGlobalNode(nn.Module):
    def __init__(self, emb_size: int, activation: nn.Module):
        super().__init__()
        self.aggr = SumAggregation()
        self.msg = MLPLayer(emb_size, emb_size, activation)
        self.linear = MLPLayer(emb_size * 2, emb_size, activation)

    def forward(self, x: Tensor, g_prev: Tensor, batch_idx: Tensor) -> Tensor:
        x = self.msg(x)
        g = self.aggr(x, batch_idx)
        return g_prev + self.linear(torch.concatenate([g, g_prev], axis=-1))


class KGGNN(nn.Module):
    def __init__(
        self,
        layers: int,
        embedding_dim: int,
        num_object_classes: int,
        num_predicate_classes: int,
        activation: nn.Module,
        aggregation: str,
    ):
        super().__init__()
        self.obj_embedding = EmbeddingLayer(
            num_object_classes, embedding_dim, activation
        )
        self.relation_embedding = EmbeddingLayer(
            num_predicate_classes, embedding_dim, activation
        )
        # self.edge_transform = MLPLayer(embedding_dim, embedding_dim, activation)
        self.convs = nn.ModuleList(
            [
                GNNConv(
                    embedding_dim,
                    embedding_dim,
                    num_predicate_classes,
                    activation,
                    aggregation,
                )
                for _ in range(layers)
            ]
        )
        self.aggrs = nn.ModuleList(
            [GlobalNode(embedding_dim, activation) for _ in range(layers)]
        )
        self.hidden_size = embedding_dim

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch_idx: Tensor
    ):
        num_graphs = batch_idx.max().item() + 1
        g = torch.zeros(num_graphs, self.hidden_size).to(x.device)
        h = self.obj_embedding(x)
        h = torch.zeros_like(h)
        h_r = self.relation_embedding(edge_attr)
        for i in range(len(self.convs)):
            h = self.convs[i](h, edge_index, h_r)
            g = self.aggrs[i](h, g, batch_idx)

        return h, g


class KGGNNAgent(nn.Module):
    def __init__(
        self,
        num_object_classes: int,
        num_predicate_classes: int,
        num_actions: int,
        layers: int,
        embedding_dim: int,
        aggregation: str,
        activation: nn.Module,
    ):
        super().__init__()
        self.p_gnn = KGGNN(
            layers,
            embedding_dim,
            num_object_classes,
            num_predicate_classes,
            activation,
            aggregation,
        )
        # self.vf_gnn = KGGNN(
        #     layers,
        #     embedding_dim,
        #     num_object_classes,
        #     num_predicate_classes,
        #     activation,
        #     aggregation,
        # )
        self.policy = GNNPolicy(num_actions, embedding_dim, activation)
        self.vf = nn.Sequential(
            MLPLayer(embedding_dim, embedding_dim, activation),
            LinearTransform(embedding_dim, 1, activation),
        )

    def value(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch_idx: Tensor,
    ):
        _, g = self.vf_gnn.forward(x, edge_index, edge_attr, batch_idx)
        value = self.vf(g)
        return value

    # differentiable action evaluation
    def forward(
        self,
        action: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch_idx: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        h, g = self.p_gnn.forward(x, edge_index, edge_attr, batch_idx)
        logprob, entropy = self.policy.forward(action, h, g, batch_idx)
        # _, g = self.vf_gnn(x, edge_index, edge_attr, batch_idx)
        value = self.vf(g)
        return logprob, entropy, value

    # sample action from policy and evaluate value
    def sample_action(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch_idx: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        h, g = self.p_gnn.forward(x, edge_index, edge_attr, batch_idx)
        actions, logprob, entropy = self.policy.sample(h, g, batch_idx)
        # _, g = self.vf_gnn(x, edge_index, edge_attr, batch_idx)
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
    gnn = KGGNNAgent(env.num_types, env.num_relations, len(env.action_values))
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
