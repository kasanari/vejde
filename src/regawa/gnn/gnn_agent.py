from dataclasses import asdict
from typing import Any, NamedTuple, TypeVar

import torch
import torch.nn as nn
from torch import Generator as Rngs
from torch import Tensor, as_tensor, concatenate, int64

from regawa.gnn.agent_utils import ActionMode, AgentConfig, GNNParams
from regawa.gnn.node_then_action import NodeThenActionPolicy

from .action_then_node import ActionThenNodePolicy
from .data import HeteroStateData, StateData
from .factorgraph_gnn import BipartiteGNN, FactorGraph
from .gnn_classes import EmbeddingLayer, SparseArray, SparseTensor, sparsify
from .gnn_embedder import BooleanEmbedder, NumericEmbedder, RecurrentEmbedder


class EmbeddedTuple(NamedTuple):
    variables: SparseTensor
    factors: SparseTensor
    globals: SparseTensor
    senders: Tensor
    receivers: Tensor
    edge_attr: Tensor
    n_variable: Tensor
    n_factor: Tensor
    action_mask: Tensor


def heterostatedata_to_tensors(
    data: HeteroStateData, device: str | torch.device = "cpu"
) -> HeteroStateData:
    return HeteroStateData(
        statedata_to_tensors(data.boolean, device),
        statedata_to_tensors(data.numeric, device),
    )


def statedata_to_tensors(
    data: StateData, device: str | torch.device = "cpu"
) -> StateData:
    params = tuple(
        SparseTensor(
            as_tensor(attr.values, device=device),
            as_tensor(attr.indices, device=device),
        )
        if isinstance(attr, SparseArray)
        else as_tensor(attr, device=device)
        for attr in data
    )

    return StateData(*params)


def _embed(
    data: StateData,
    var_embedder: nn.Module,
    factor_embedding: nn.Module,
    global_var_embedder: nn.Module,
    edge_attr_emb: nn.Module,
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
        else SparseTensor(
            as_tensor([], device=data.global_vals.values.device),
            as_tensor([], dtype=int64, device=data.global_vals.values.device),
        )
    )

    embedd_edge_attr = edge_attr_emb(data.edge_attr)

    return EmbeddedTuple(
        variables,
        factors,
        globals_,
        data.senders,
        data.receivers,
        embedd_edge_attr,
        data.n_variable,
        data.n_factor,
        data.action_mask,
    )


def cat(a: Tensor, b: Tensor) -> Tensor:
    return concatenate((a, b))


@torch.jit.script
def concat_sparse(a: SparseTensor, b: SparseTensor) -> SparseTensor:
    return SparseTensor(
        concatenate((a.values, b.values)),
        concatenate((a.indices, b.indices)),
    )


@torch.jit.script
def merge_graphs(
    boolean: EmbeddedTuple,
    numeric: EmbeddedTuple,
) -> FactorGraph:
    # this only refers to the factors, so we can use either boolean or numeric data

    return FactorGraph(
        concat_sparse(boolean.variables, numeric.variables),
        # same factors for both boolean and numeric data, so we can use either
        boolean.factors,
        cat(boolean.senders, numeric.senders + sum(boolean.n_variable)),
        # since factors are the same, we do not need to offset the receiver indices
        cat(boolean.receivers, numeric.receivers),
        cat(boolean.edge_attr, numeric.edge_attr),
        boolean.n_factor,
        concat_sparse(boolean.globals, numeric.globals),
        boolean.action_mask,
    )


class GraphAgent(nn.Module):
    def __init__(
        self,
        config: AgentConfig,
        rngs: Rngs,
    ):
        super().__init__()  # type: ignore

        self.config = config
        gnn_params = config.hyper_params

        self.factor_embedding = EmbeddingLayer(
            config.num_object_classes, gnn_params.embedding_dim, rngs
        )

        self.predicate_embedding = EmbeddingLayer(
            config.num_predicate_classes, gnn_params.embedding_dim, rngs
        )

        self.edge_attr_embedding = EmbeddingLayer(
            config.arity, gnn_params.embedding_dim, rngs, use_padding=False
        )

        self.boolean_embedder = BooleanEmbedder(
            gnn_params.embedding_dim,
            self.predicate_embedding,
            rngs,
        )

        self.numeric_embedder = NumericEmbedder(
            gnn_params.embedding_dim,
            gnn_params.activation,
            self.predicate_embedding,
        )

        self.p_gnn = BipartiteGNN(
            gnn_params.layers,
            gnn_params.embedding_dim,
            gnn_params.aggregation,
            gnn_params.activation,
            rngs,
        )

        policy_args = (config.num_actions, gnn_params.embedding_dim, rngs)
        self.policy = (
            ActionThenNodePolicy(*policy_args)
            if gnn_params.action_mode == ActionMode.ACTION_THEN_NODE
            else NodeThenActionPolicy(*policy_args)
        )

    def embed(self, data: HeteroStateData) -> FactorGraph:
        return self.p_gnn(
            merge_graphs(
                _embed(
                    data.boolean,
                    self.boolean_embedder,
                    self.factor_embedding,
                    self.boolean_embedder,
                    self.edge_attr_embedding,
                ),
                _embed(
                    data.numeric,
                    self.numeric_embedder,
                    self.factor_embedding,
                    self.numeric_embedder,
                    self.edge_attr_embedding,
                ),
            )
        )

    def forward(self, actions: Tensor, data: HeteroStateData):
        fg = self.embed(data)
        return self.policy(actions, fg.factors, fg.action_mask, fg.n_factor)

    def sample(self, data: HeteroStateData, deterministic: bool = False):
        fg = self.embed(data)
        return self.policy.sample(
            fg.factors, fg.n_factor, fg.action_mask, deterministic
        )

    def value(self, data: HeteroStateData):
        fg = self.embed(data)
        return self.policy.value(fg.factors, fg.n_factor, fg.action_mask)

    def save_agent(self, path: str):
        save_agent(self, self.config, path)

    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def load_agent(cls, path: str) -> tuple["GraphAgent", AgentConfig]:
        return load_agent(cls, path)  # type: ignore


class RecurrentGraphAgent(nn.Module):
    def __init__(
        self,
        config: AgentConfig,
        rngs: Rngs,
    ):
        super().__init__()  # type: ignore

        self.config = config
        gnn_params = config.hyper_params

        self.factor_embedding = EmbeddingLayer(
            config.num_object_classes,
            gnn_params.embedding_dim,
            rngs,
        )

        self.predicate_embedding = EmbeddingLayer(
            config.num_predicate_classes,
            gnn_params.embedding_dim,
            rngs,
        )

        boolean_embedder = BooleanEmbedder(
            gnn_params.embedding_dim,
            self.predicate_embedding,
        )

        self.r_boolean_embedder = RecurrentEmbedder(
            gnn_params.embedding_dim,
            boolean_embedder,
        )

        self.edge_attr_embedding = EmbeddingLayer(
            config.arity, gnn_params.embedding_dim, rngs, use_padding=False
        )

        numeric_embedder = NumericEmbedder(
            gnn_params.embedding_dim,
            gnn_params.activation,
            self.predicate_embedding,
        )

        self.r_numeric_embedder = RecurrentEmbedder(
            gnn_params.embedding_dim,
            numeric_embedder,
        )

        self.p_gnn = BipartiteGNN(
            gnn_params.layers,
            gnn_params.embedding_dim,
            gnn_params.aggregation,
            gnn_params.activation,
        )

        policy_args = (config.num_actions, gnn_params.embedding_dim)
        self.policy = (
            ActionThenNodePolicy(*policy_args)
            if gnn_params.action_mode == ActionMode.ACTION_THEN_NODE
            else NodeThenActionPolicy(*policy_args)
        )

    def embed(self, data: HeteroStateData) -> FactorGraph:
        return self.p_gnn(
            merge_graphs(
                _embed(
                    data.boolean,
                    self.r_boolean_embedder(data.boolean.length),
                    self.factor_embedding,
                    self.r_boolean_embedder(data.boolean.global_length),
                    self.edge_attr_embedding,
                ),
                _embed(
                    data.numeric,
                    self.r_numeric_embedder(data.numeric.length),
                    self.factor_embedding,
                    self.r_numeric_embedder(data.numeric.global_length),
                    self.edge_attr_embedding,
                ),
            )
        )

    def forward(self, actions: Tensor, data: HeteroStateData):
        fg = self.embed(data)
        return self.policy(actions, fg.factors, fg.action_mask, fg.n_factor)

    def sample(self, data: HeteroStateData, deterministic: bool = False):
        fg = self.embed(data)
        return self.policy.sample(
            fg.factors, fg.n_factor, fg.action_mask, deterministic
        )

    def value(self, data: HeteroStateData):
        fg = self.embed(data)
        _, _, _, value, *_ = self.policy.sample(
            fg.factors, fg.n_factor, fg.action_mask, False
        )
        return value

    def save_agent(self, path: str):
        save_agent(self, self.config, path)

    @classmethod
    def load_agent(cls, path: str) -> tuple["RecurrentGraphAgent", AgentConfig]:
        return load_agent(cls, path)  # type: ignore


def save_agent(agent: GraphAgent | RecurrentGraphAgent, config: AgentConfig, path: str):
    state_dict = agent.state_dict()
    to_save: dict[str, Any] = {}
    to_save["config"] = asdict(config)
    to_save["state_dict"] = state_dict
    torch.save(to_save, path)  # type: ignore


T = TypeVar("T", bound=GraphAgent | RecurrentGraphAgent)


def load_agent(cls: T, path: str) -> tuple[T, AgentConfig]:
    data = torch.load(path, weights_only=False)  # type: ignore

    data["config"]["hyper_params"] = GNNParams(**data["config"]["hyper_params"])

    config = AgentConfig(**data["config"])
    agent = cls(config)
    agent.load_state_dict(data["state_dict"])

    return agent, config
