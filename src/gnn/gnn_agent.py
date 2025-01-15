from dataclasses import dataclass
from typing import Any, NamedTuple, TypeVar

import torch.nn as nn
from torch import Tensor

import torch as th
from dataclasses import asdict

from gnn.gnn_classes import EmbeddingLayer
from .data import HeteroStateData, StateData
from .factorgraph_gnn import BatchIdx, BipartiteGNN, FactorGraph
from .gnn_embedder import (
    BooleanEmbedder,
    NumericEmbedder,
    RecurrentEmbedder,
)
from .gnn_policies import ActionMode, TwoActionGNNPolicy


class EmbeddedTuple(NamedTuple):
    variables: Tensor
    factors: Tensor
    globals: Tensor


def _embed(
    data: StateData,
    var_embedder: nn.Module,
    factor_embedding: nn.Module,
    global_var_embedder: nn.Module,
) -> EmbeddedTuple:
    factors = factor_embedding(data.factor)
    variables = var_embedder(
        data.var_value,
        data.var_type,
    )
    globals_ = (
        global_var_embedder(
            data.global_vals,
            data.global_vars,
        )
        if data.global_vals.shape[0] > 0
        else th.tensor([])
    )
    return EmbeddedTuple(variables, factors, globals_)


def cat(a: Tensor, b: Tensor) -> Tensor:
    return th.cat((a, b))


def merge_graphs(
    boolean_data: StateData,
    numeric_data: StateData,
    boolean: EmbeddedTuple,
    numeric: EmbeddedTuple,
) -> FactorGraph:
    # this only refers to the factors, so we can use either boolean or numeric data
    factor_batch = boolean_data.factor_batch

    global_batch = cat(
        boolean_data.global_batch,
        numeric_data.global_batch,
    )

    variable_batch_idx = cat(
        boolean_data.var_batch,
        numeric_data.var_batch,
    )

    return FactorGraph(
        cat(boolean.variables, numeric.variables),
        # same factors for both boolean and numeric data, so we can use either
        boolean.factors,
        cat(boolean_data.senders, numeric_data.senders + sum(boolean_data.n_variable)),
        # since factors are the same, we do not need to offset the receiver indices
        cat(boolean_data.receivers, numeric_data.receivers),
        cat(boolean_data.edge_attr, numeric_data.edge_attr),
        boolean_data.n_factor,
        cat(boolean.globals, numeric.globals),
        BatchIdx(global_batch, factor_batch, variable_batch_idx),
    )


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
        super().__init__()  # type: ignore

        self.config = config

        self.factor_embedding = EmbeddingLayer(
            config.num_object_classes, config.embedding_dim
        )

        self.predicate_embedding = EmbeddingLayer(
            config.num_predicate_classes, config.embedding_dim
        )

        self.boolean_embedder = BooleanEmbedder(
            config.embedding_dim,
            self.predicate_embedding,
        )

        self.numeric_embedder = NumericEmbedder(
            config.embedding_dim,
            config.activation,
            self.predicate_embedding,
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

    def embed(self, data: HeteroStateData) -> tuple[FactorGraph, Tensor]:
        numeric_data = data.numeric
        boolean_data = data.boolean

        fg = merge_graphs(
            boolean_data,
            numeric_data,
            _embed(
                boolean_data,
                self.boolean_embedder,
                self.factor_embedding,
                self.boolean_embedder,
            ),
            _embed(
                numeric_data,
                self.numeric_embedder,
                self.factor_embedding,
                self.numeric_embedder,
            ),
        )
        e_fg = self.p_gnn(fg)
        return e_fg

    def forward(self, actions: Tensor, data: HeteroStateData):
        fg, g = self.embed(data)
        logprob, entropy, value = self.actorcritic(
            actions, fg.factors, g, fg.batch.factor, fg.n_factor
        )
        return logprob, entropy, value

    def sample(self, data: HeteroStateData, deterministic: bool = False):
        fg, g = self.embed(data)
        action, logprob, entropy = self.actorcritic.sample(
            fg.factors, g, fg.batch.factor, fg.n_factor, deterministic
        )
        value = self.actorcritic.value(g)
        return action, logprob, entropy, value

    def value(self, data: HeteroStateData):
        _, g = self.embed(data)
        return self.actorcritic.value(g)

    def save_agent(self, path: str):
        save_agent(self, self.config, path)

    @classmethod
    def load_agent(cls, path: str) -> tuple["RecurrentGraphAgent", Config]:
        return load_agent(cls, path)  # type: ignore


class RecurrentGraphAgent(nn.Module):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()  # type: ignore

        self.config = config

        self.factor_embedding = EmbeddingLayer(
            config.num_object_classes,
            config.embedding_dim,
        )

        self.predicate_embedding = EmbeddingLayer(
            config.num_predicate_classes,
            config.embedding_dim,
        )

        boolean_embedder = BooleanEmbedder(
            config.embedding_dim,
            self.predicate_embedding,
        )

        self.r_boolean_embedder = RecurrentEmbedder(
            config.embedding_dim,
            boolean_embedder,
        )

        numeric_embedder = NumericEmbedder(
            config.embedding_dim,
            config.activation,
            self.predicate_embedding,
        )

        self.r_numeric_embedder = RecurrentEmbedder(
            config.embedding_dim,
            numeric_embedder,
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

    def embed(self, data: HeteroStateData) -> tuple[FactorGraph, Tensor]:
        fg = merge_graphs(
            data.boolean,
            data.numeric,
            _embed(
                data.boolean,
                self.r_boolean_embedder(data.boolean.length),
                self.factor_embedding,
                self.r_boolean_embedder(data.boolean.global_length),
            ),
            _embed(
                data.numeric,
                self.r_numeric_embedder(data.numeric.length),
                self.factor_embedding,
                self.r_numeric_embedder(data.numeric.global_length),
            ),
        )
        e_fg, g = self.p_gnn(fg)
        return e_fg, g

    def forward(self, actions: Tensor, data: HeteroStateData):
        fg, g = self.embed(data)
        return self.actorcritic(actions, fg.factors, g, fg.batch.factor, fg.n_factor)

    def sample(self, data: HeteroStateData, deterministic: bool = False):
        fg, g = self.embed(data)
        action, logprob, entropy = self.actorcritic.sample(
            fg.factors, g, fg.batch.factor, fg.n_factor, deterministic
        )
        value = self.actorcritic.value(g)
        return action, logprob, entropy, value

    def value(self, data: HeteroStateData):
        _, g = self.embed(data)
        return self.actorcritic.value(g)

    def save_agent(self, path: str):
        save_agent(self, self.config, path)

    @classmethod
    def load_agent(cls, path: str) -> tuple["RecurrentGraphAgent", Config]:
        return load_agent(cls, path)  # type: ignore


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
        n_nodes: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return (
            *self.policy.forward(actions, h, g, batch_idx, n_nodes),
            self.vf(g),
        )

    def sample(
        self,
        h: Tensor,
        g: Tensor,
        batch_idx: Tensor,
        n_nodes: Tensor,
        deterministic: bool = False,
    ):
        return self.policy.sample(h, g, batch_idx, n_nodes, deterministic)


def save_agent(agent: GraphAgent | RecurrentGraphAgent, config: Config, path: str):
    state_dict = agent.state_dict()
    to_save: dict[str, Any] = {}
    to_save["config"] = asdict(config)
    to_save["state_dict"] = state_dict
    th.save(to_save, path)  # type: ignore


T = TypeVar("T", bound=GraphAgent | RecurrentGraphAgent)


def load_agent(cls: T, path: str) -> tuple[T, Config]:
    data = th.load(path, weights_only=False)  # type: ignore
    config = Config(**data["config"])
    agent = cls(config)
    agent.load_state_dict(data["state_dict"])

    return agent, config
