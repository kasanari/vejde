from .factorgraph_gnn import BipartiteGNN, FactorGraph, FactorGraphEmbedding
from .gnn_embedder import Embedder, RecurrentEmbedder
from .gnn_policies import ActionMode, TwoActionGNNPolicy
from .data import StackedStateData, StateData
from torch import Tensor
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class Config:
    num_object_classes: int
    num_predicate_classes: int
    num_actions: int
    embedding_dim: int
    layers: int
    aggregation: str
    activation: nn.Module
    action_mode: ActionMode
    recurrent = False


class GraphAgent(nn.Module):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()
        self.embedder = Embedder(
            config.num_object_classes,
            config.num_predicate_classes,
            config.embedding_dim,
            config.activation,
        )

        self.p_gnn = BipartiteGNN(
            config.layers,
            config.embedding_dim,
            config.aggregation,
            config.activation,
        )

        self.actorcritic = GraphActorCritic(
            config.num_actions,
            config.embedding_dim,
            config.action_mode,
        )

    def embed(self, data: StateData) -> FactorGraphEmbedding:
        variables, factors = self.embedder(
            data.var_val, data.var_type, data.object_class
        )
        fg = FactorGraph(
            variables,
            factors,
            data.edge_index,
            data.edge_attr,
            data.batch_idx,
        )
        e_fg = self.p_gnn(fg)
        return e_fg

    def forward(self, actions: Tensor, data: StateData):
        e_fg = self.embed(data)
        logprob, entropy, value = self.actorcritic(
            actions, e_fg.factors, e_fg.graph, data.batch_idx
        )
        return logprob, entropy, value

    def sample(self, data: StateData, deterministic: bool = False):
        e_fg = self.embed(data)
        action, logprob, entropy = self.actorcritic.sample(
            e_fg.factors, e_fg.graph, data.batch_idx, deterministic
        )
        value = self.actorcritic.value(e_fg.graph)
        return action, logprob, entropy, value

    def value(self, data: StateData):
        e_fg = self.embed(data)
        return self.actorcritic.value(e_fg.graph)


class RecurrentGraphAgent(nn.Module):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()
        self.embedder = RecurrentEmbedder(
            config.num_object_classes,
            config.num_predicate_classes,
            config.embedding_dim,
            config.activation,
        )

        self.p_gnn = BipartiteGNN(
            config.layers,
            config.embedding_dim,
            config.aggregation,
            config.activation,
        )

        self.actorcritic = GraphActorCritic(
            config.num_actions,
            config.embedding_dim,
            config.action_mode,
        )

    def embed(self, data: StackedStateData) -> FactorGraphEmbedding:
        variables, factors = self.embedder(
            data.var_val, data.var_type, data.object_class, data.lengths
        )
        e_fg = self.p_gnn(
            FactorGraph(
                variables,
                factors,
                data.edge_index,
                data.edge_attr,
                data.batch_idx,
            )
        )
        return e_fg

    def forward(self, actions: Tensor, data: StackedStateData):
        e_fg = self.embed(data)
        return self.actorcritic(actions, e_fg.factors, e_fg.graph, data.batch_idx)

    def sample(self, data: StackedStateData, deterministic: bool = False):
        e_fg = self.embed(data)
        return self.actorcritic.sample(
            e_fg.factors, e_fg.graph, data.batch_idx, deterministic
        )


class GraphActorCritic(nn.Module):
    def __init__(
        self,
        num_actions: int,
        embedding_dim: int,
        action_mode: ActionMode,
    ):
        super().__init__()  # type: ignore
        self.policy = TwoActionGNNPolicy(num_actions, embedding_dim, action_mode)
        self.vf = nn.Linear(embedding_dim, 1)

    def value(
        self,
        g: Tensor,
    ):
        return self.vf(g)

    def forward(
        self,
        actions: Tensor,
        h: Tensor,
        g: Tensor,
        batch_idx: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return (
            *self.policy.forward(actions, h, g, batch_idx),
            self.vf(g),
        )

    def sample(
        self,
        h: Tensor,
        g: Tensor,
        batch_idx: Tensor,
        deterministic: bool = False,
    ):
        return self.policy.sample(h, g, batch_idx, deterministic)
