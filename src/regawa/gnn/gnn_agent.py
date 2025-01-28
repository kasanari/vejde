from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple, TypeVar

import torch.nn as nn
from torch import Tensor

import torch as th
from dataclasses import asdict

from regawa.gnn.node_then_action import NodeThenActionPolicy

from .gnn_classes import EmbeddingLayer, SparseTensor, sparsify
from .data import HeteroStateData, StateData
from .factorgraph_gnn import BipartiteGNN, FactorGraph
from .gnn_embedder import (
    BooleanEmbedder,
    NumericEmbedder,
    RecurrentEmbedder,
)

from .action_then_node import ActionThenNodePolicy


class ActionMode(Enum):
    ACTION_THEN_NODE = 0
    NODE_THEN_ACTION = 1
    ACTION_AND_NODE = 2


class EmbeddedTuple(NamedTuple):
    variables: SparseTensor
    factors: SparseTensor
    globals: SparseTensor


def _embed(
    data: StateData,
    var_embedder: nn.Module,
    factor_embedding: nn.Module,
    global_var_embedder: nn.Module,
) -> EmbeddedTuple:
    factors = sparsify(factor_embedding)(data.factor)
    variables = SparseTensor(
        var_embedder(
            data.var_value.values,
            data.var_type.values,
        ),
        data.var_value.indices,
    )
    globals_ = (
        SparseTensor(
            global_var_embedder(
                data.global_vals.values,
                data.global_vars.values,
            ),
            data.global_vals.indices,
        )
        if data.global_vals.shape[0] > 0
        else SparseTensor(th.tensor([]), th.tensor([], dtype=th.long))
    )
    return EmbeddedTuple(variables, factors, globals_)


def cat(a: Tensor, b: Tensor) -> Tensor:
    return th.cat((a, b))


@th.jit.script
def concat_sparse(a: SparseTensor, b: SparseTensor) -> SparseTensor:
    return SparseTensor(
        th.cat((a.values, b.values)),
        th.cat((a.indices, b.indices)),
    )


@th.jit.script
def merge_graphs(
    boolean_data: StateData,
    numeric_data: StateData,
    boolean: EmbeddedTuple,
    numeric: EmbeddedTuple,
) -> FactorGraph:
    # this only refers to the factors, so we can use either boolean or numeric data

    return FactorGraph(
        concat_sparse(boolean.variables, numeric.variables),
        # same factors for both boolean and numeric data, so we can use either
        boolean.factors,
        cat(boolean_data.senders, numeric_data.senders + sum(boolean_data.n_variable)),
        # since factors are the same, we do not need to offset the receiver indices
        cat(boolean_data.receivers, numeric_data.receivers),
        cat(boolean_data.edge_attr, numeric_data.edge_attr),
        boolean_data.n_factor,
        concat_sparse(boolean.globals, numeric.globals),
        boolean_data.action_mask,
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
            config.embedding_dim,
            config.action_mode,
        )

    def embed(self, data: HeteroStateData) -> FactorGraph:
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
        fg = self.embed(data)
        logprob, entropy, value = self.actorcritic(
            actions, fg.factors, fg.action_mask, fg.n_factor
        )
        return logprob, entropy, value

    def sample(self, data: HeteroStateData, deterministic: bool = False):
        fg = self.embed(data)
        action, logprob, entropy, value = self.actorcritic.sample(
            fg.factors, fg.n_factor, fg.action_mask, deterministic
        )
        return action, logprob, entropy, value

    def value(self, data: HeteroStateData):
        fg = self.embed(data)
        _, _, _, value = self.actorcritic.sample(
            fg.factors, fg.n_factor, fg.action_mask, False
        )
        return value

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
            config.embedding_dim,
            config.action_mode,
        )

    def embed(self, data: HeteroStateData) -> FactorGraph:
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
        e_fg = self.p_gnn(fg)
        return e_fg

    def forward(self, actions: Tensor, data: HeteroStateData):
        fg = self.embed(data)
        return self.actorcritic(actions, fg.factors, fg.action_mask, fg.n_factor)

    def sample(self, data: HeteroStateData, deterministic: bool = False):
        fg = self.embed(data)
        action, logprob, entropy, value = self.actorcritic.sample(
            fg.factors, fg.n_factor, fg.action_mask, deterministic
        )
        return action, logprob, entropy, value

    def value(self, data: HeteroStateData):
        fg = self.embed(data)
        _, _, _, value = self.actorcritic.sample(
            fg.factors, fg.n_factor, fg.action_mask, False
        )
        return value

    def save_agent(self, path: str):
        save_agent(self, self.config, path)

    @classmethod
    def load_agent(cls, path: str) -> tuple["RecurrentGraphAgent", Config]:
        return load_agent(cls, path)  # type: ignore


class GraphActorCritic(nn.Module):
    def __init__(
        self,
        num_actions: int,
        node_dim: int,
        graph_dim: int,
        action_mode: ActionMode,
    ):
        super().__init__()  # type: ignore
        self.policy = (
            ActionThenNodePolicy(num_actions, node_dim)
            if action_mode == ActionMode.ACTION_THEN_NODE
            else NodeThenActionPolicy(num_actions, node_dim)
        )
        self.num_actions = num_actions

    def forward(
        self,
        actions: Tensor,
        h: SparseTensor,
        action_mask: Tensor,
        n_nodes: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.policy.forward(actions, h, action_mask, n_nodes)

    def sample(
        self,
        h: SparseTensor,
        n_nodes: Tensor,
        action_mask: Tensor,
        deterministic: bool = False,
    ):
        return self.policy.sample(h, n_nodes, action_mask, deterministic)


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
