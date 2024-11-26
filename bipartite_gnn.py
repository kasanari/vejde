from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
import torch
from gnn_policies import ActionMode, TwoActionGNNPolicy
from wrappers.kg_wrapper import KGRDDLGraphWrapper
from torch import Tensor, LongTensor
import torch.nn as nn
import random
import numpy as np
from numpy.typing import NDArray

import logging

from torch_geometric.nn import AttentionalAggregation, GINConv, SumAggregation
from torch_geometric.utils import to_undirected

from wrappers.wrapper import GroundedRDDLGraphWrapper


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class BipartiteData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return torch.tensor([[self.x_p.size(0)], [self.x_o.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)


class EmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, activation: nn.Module):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # nn.init.orthogonal_(self.embedding.weight)
        # nn.init.ones_(self.embedding.weight)

        # self.bias = nn.Parameter(torch.zeros(num_embeddings, embedding_dim))
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        # bias = self.bias[x]
        return self.embedding(x)


class MLPLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: nn.Module):
        super(MLPLayer, self).__init__()
        self.linear = layer_init(nn.Linear(in_features, out_features))
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.linear(x))


class BipartiteGNNConvVariableToFactor(MessagePassing):
    def __init__(
        self,
        aggr: str,
        in_channels: int,
        out_channels: int,
        activation: nn.Module,
    ):
        super().__init__(aggr=aggr, flow="source_to_target")
        self.root = MLPLayer(in_channels, out_channels, activation)
        self.combine = MLPLayer(out_channels * 2, out_channels, activation)
        self.message_func = MLPLayer(in_channels * 2, out_channels, activation)
        # self.relation_embedding = nn.Embedding(num_relations, in_channels)
        # nn.init.orthogonal_(self.relation_embedding.weight)

    def forward(
        self,
        var_val: Tensor,
        factor: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ):
        """
        x_p: Tensor of shape (num_predicates, embedding_dim),
        x_o: Tensor of shape (num_objects, embedding_dim),
        edge_index: Tensor of shape (2, num_edges),
        edge_attr: Tensor of shape (num_edges,)
        """

        # fac2var_edges = edge_index.flip(0)

        # fac2var_messages = self.edge_updater(
        #     x=(factor, var_val),
        #     edge_index=fac2var_edges,
        #     size=(
        #         factor.size(0),
        #         var_val.size(0),
        #     ),
        # )

        return self.propagate(
            edge_index,
            x=(var_val, factor),
            edge_attr=edge_attr,
            # fac2var_messages=fac2var_messages,
            size=(var_val.size(0), factor.size(0)),
        )

    def edge_update(self, x_j):
        return x_j

    def message(
        self,
        x_j: Tensor,
        x_i: Tensor,
        # fac2var_messages: Tensor,
    ) -> Tensor:
        # x_j = x_j - fac2var_messages
        return self.message_func(torch.concatenate([x_i, x_j], dim=-1))

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        _, x_o = x
        new = self.combine(torch.concatenate([x_o, aggr_out], dim=-1))
        return new


class BipartiteGNNConvFactorToVariable(MessagePassing):
    def __init__(
        self, aggr: str, in_channels: int, out_channels: int, activation: nn.Module
    ):
        super().__init__(aggr=aggr, flow="source_to_target")
        # self.root = MLPLayer(in_channels, out_channels, activation)
        self.combine = MLPLayer(out_channels * 2, out_channels, activation)
        self.message_func = MLPLayer(in_channels * 2, out_channels, activation)
        # self.messages = None
        # self.relation_embedding = nn.Embedding(num_relations, in_channels)
        # nn.init.orthogonal_(self.relation_embedding.weight)

    def forward(
        self,
        variables: Tensor,
        factors: Tensor,
        edge_index: Tensor,  # edges are (var, factor)
        edge_attr: Tensor,
    ):
        """
        x_p: Tensor of shape (num_predicates, embedding_dim),
        x_o: Tensor of shape (num_objects, embedding_dim),
        edge_index: Tensor of shape (2, num_edges),
        edge_attr: Tensor of shape (num_edges,)
        """

        fac2var_edges = edge_index.flip(0)

        return self.propagate(
            x=(factors, variables),
            edge_index=fac2var_edges,
            # edge_attr=edge_attr,
            size=(
                factors.size(0),
                variables.size(0),
            ),
        )

    def message(
        self,
        x_j: Tensor,
        x_i: Tensor,
    ) -> Tensor:  # type: ignore
        return self.message_func(torch.concatenate([x_i, x_j], dim=-1))

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:  # type: ignore
        _, x_p = x
        new = x_p + self.combine(torch.concatenate([x_p, aggr_out], dim=-1))
        return new


class GlobalNode(nn.Module):
    def __init__(self, emb_size: int, activation: nn.Module):
        super().__init__()  # type: ignore
        self.aggr = AttentionalAggregation(
            nn.Linear(emb_size, 1), nn.Linear(emb_size, emb_size)
        )
        self.linear = MLPLayer(emb_size * 2, emb_size, activation)
        # self.aggr = SumAggregation()

    def forward(self, x: Tensor, g_prev: Tensor, batch_idx: Tensor) -> Tensor:
        g = self.aggr(x, batch_idx)
        return g_prev + self.linear(torch.concatenate([g, g_prev], dim=-1))


class FactorGraphLayer(nn.Module):
    def __init__(self, embedding_dim: int, aggregation: str, activation: nn.Module):
        super().__init__()
        self.var2factor = BipartiteGNNConvVariableToFactor(
            aggregation, embedding_dim, embedding_dim, activation
        )

        self.factor2var = BipartiteGNNConvFactorToVariable(
            aggregation, embedding_dim, embedding_dim, activation
        )

        # self.factor2var = GINConv(
        #     nn.Sequential(
        #         nn.Linear(embedding_dim, embedding_dim),
        #         nn.ReLU(),
        #     ),
        #     train_eps=True,
        # )

    def forward(
        self,
        h_p: Tensor,
        h_o: Tensor,
        edge_index: Tensor,  # edges are (var, factor)
        edge_attr: Tensor,
    ) -> tuple[Tensor, Tensor]:
        n_h_o = self.var2factor(
            h_p,
            h_o,
            edge_index,
            edge_attr,
        )

        n_h_p = self.factor2var(
            h_p,
            h_o,
            edge_index,
            edge_attr,
        )

        return n_h_p, n_h_o


class BipartiteGNN(nn.Module):
    def __init__(
        self,
        layers: int,
        embedding_dim: int,
        num_object_classes: int,
        num_predicate_classes: int,
        aggregation: str,
        activation: nn.Module,
    ):
        super().__init__()  # type: ignore
        self.obj_embedding = EmbeddingLayer(
            num_object_classes, embedding_dim, activation
        )
        self.predicate_embedding = EmbeddingLayer(
            num_predicate_classes, embedding_dim, activation
        )

        self.convs = nn.ModuleList(
            [
                FactorGraphLayer(embedding_dim, aggregation, activation)
                for _ in range(layers)
            ]
        )

        # self.aggrs = [GlobalNode(embedding_dim, activation) for _ in range(layers)]
        self.aggr = GlobalNode(embedding_dim, activation)
        self.hidden_size = embedding_dim

    def forward(
        self,
        grounding_value: Tensor,
        grounding_class: Tensor,
        object_class: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch_idx: Tensor,
    ) -> tuple[Tensor, Tensor]:
        h_p, h_o = self.embed_nodes(grounding_value, grounding_class, object_class)

        h_o, g = self.embed_graph(h_p, h_o, edge_index, edge_attr, batch_idx)

        return h_o, g

    def embed_nodes(
        self, grounding_value: Tensor, grounding_class: Tensor, object_class: Tensor
    ) -> tuple[Tensor, Tensor]:
        h_c = self.predicate_embedding(grounding_class)
        h_p = grounding_value.unsqueeze(-1).float() * h_c
        h_o = self.obj_embedding(object_class)

        return h_p, h_o

    def embed_graph(
        self,
        variables: Tensor,
        factors: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch_idx: Tensor,
    ) -> tuple[Tensor, Tensor]:
        num_graphs = int(batch_idx.max().item() + 1)
        g = torch.zeros(num_graphs, self.hidden_size).to(factors.device)

        for conv in self.convs:
            (variables, factors) = conv(variables, factors, edge_index, edge_attr)

        g = self.aggr(factors, g, batch_idx)

        return factors, g


class BipartiteGNNAgent(nn.Module):
    def __init__(
        self,
        num_object_classes: int,
        num_predicate_classes: int,
        num_actions: int,
        layers: int,
        embedding_dim: int,
        activation: nn.Module,
        aggregation: str,
        action_mode: ActionMode,
    ):
        super().__init__()  # type: ignore
        self.p_gnn = BipartiteGNN(
            layers,
            embedding_dim,
            num_object_classes,
            num_predicate_classes,
            aggregation,
            activation,
        )
        # self.vf_gnn = BipartiteGNN(
        #     layers,
        #     embedding_dim,
        #     num_object_classes,
        #     num_predicate_classes,
        #     aggregation,
        #     activation,
        # )
        self.policy = TwoActionGNNPolicy(num_actions, embedding_dim, action_mode)
        self.vf = nn.Linear(embedding_dim, 1)

    def value(
        self,
        x_p: Tensor,
        x_o: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch_idx: Batch,
        num_graphs: int,
    ):
        _, g = self.vf_gnn(x, edge_index, edge_attr, batch_idx)
        value = self.vf(g)
        return value

    def forward(
        self,
        actions: Tensor,
        var_val: Tensor,
        var_type: Tensor,
        factor_type: Tensor,
        edge_index: Tensor,  # edges are (var, factor)
        edge_attr: Tensor,
        batch_idx: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        h, g = self.p_gnn(
            var_val, var_type, factor_type, edge_index, edge_attr, batch_idx
        )

        # split_list, data_starts = data_splits_and_starts(batch_idx)

        # h_split = torch.split(h, split_list)
        # batch_idx_split = torch.split(batch_idx, split_list)
        # new_hs = torch.concat(
        #     [
        #         torch.concat([torch.atleast_2d(g[i]), h_split[i]])
        #         for i in range(len(h_split))
        #     ],
        #     dim=0,
        # )
        # new_batch_index = torch.concat(
        #     [
        #         torch.concat([batch_idx_split[i], torch.ones(1, dtype=torch.int64) * i])
        #         for i in range(len(batch_idx_split))
        #     ],
        #     dim=0,
        # )

        logprob, entropy = self.policy.forward(actions, h, g, batch_idx)

        # _, g = self.vf_gnn(
        #     var_val, var_type, factor_type, edge_index, edge_attr, batch_idx
        # )
        value = self.vf(g)
        return logprob, entropy, value

    def sample(
        self,
        var_val: Tensor,
        var_type: Tensor,
        factor_type: Tensor,
        edge_index: Tensor,  # edges are (var, factor)
        edge_attr: Tensor,
        batch_idx: Tensor,
        deterministic: bool = False,
    ):
        h, g = self.p_gnn(
            var_val, var_type, factor_type, edge_index, edge_attr, batch_idx
        )
        action, logprob, entropy = self.policy.sample(h, g, batch_idx, deterministic)
        return action, logprob, entropy


def obs_to_data(obs: dict[str, NDArray[np.int64]]) -> Data:
    return BipartiteData(
        x_p=LongTensor(obs["predicate_value"]),
        x_c=LongTensor(obs["predicate_class"]),
        x_o=LongTensor(obs["object"]),
        edge_index=LongTensor(obs["edge_index"]).T,
        edge_attr=LongTensor(obs["edge_attr"]),
        num_nodes=obs["predicate_value"].shape[0] + obs["object"].shape[0],
    )


if __name__ == "__main__":
    main()
