from dataclasses import asdict
from typing import Any, Mapping, TypeVar

import torch
import torch.nn as nn
from torch import Generator as Rngs
from torch import Tensor, as_tensor, concatenate, int64

from regawa.gnn.agent_utils import ActionMode, AgentConfig, GNNParams
from regawa.gnn.node_then_action import NodeThenActionPolicy
from regawa.model.base_model import BaseModel

from .action_then_node import ActionThenNodePolicy
from .data import (
    FactorGraph,
    HeteroBatchData,
    ObsData,
    BatchData,
    single_obs_to_heterostatedata,
)
from .factorgraph_gnn import BipartiteGNN
from .gnn_classes import EmbeddingLayer, SparseArray, SparseTensor, sparsify
from .gnn_embedder import (
    BooleanEmbedder,
    NegativeBiasBooleanEmbedder,
    NumericEmbedder,
    RecurrentEmbedder,
)


def heterostatedata_to_tensors(
    data: HeteroBatchData, device: str | torch.device = "cpu"
) -> HeteroBatchData:
    return HeteroBatchData(
        statedata_to_tensors(data.boolean, device),
        statedata_to_tensors(data.numeric, device),
    )


def statedata_to_tensors(
    data: BatchData, device: str | torch.device = "cpu"
) -> BatchData:
    params = tuple(
        SparseTensor(
            as_tensor(attr.values, device=device),
            as_tensor(attr.indices, device=device),
        )
        if isinstance(attr, SparseArray)
        else as_tensor(attr, device=device)
        for attr in data
    )

    return BatchData(*params)


def _embed(
    data: BatchData,
    var_embedder: nn.Module,
    factor_embedding: nn.Module,
    global_var_embedder: nn.Module,
    edge_attr_emb: nn.Module,
) -> FactorGraph:
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

    return FactorGraph(
        variables,
        factors,
        globals_,
        data.senders,
        data.receivers,
        embedd_edge_attr,
        data.n_variable,
        data.n_factor,
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
    boolean: FactorGraph,
    numeric: FactorGraph,
) -> FactorGraph:
    # this only refers to the factors, so we can use either boolean or numeric data

    return FactorGraph(
        concat_sparse(boolean.variables, numeric.variables),
        # same factors for both boolean and numeric data, so we can use either
        boolean.factors,
        concat_sparse(boolean.globals, numeric.globals),
        cat(boolean.v_to_f, numeric.v_to_f + sum(boolean.n_variable)),
        # since factors are the same, we do not need to offset the receiver indices
        cat(boolean.f_to_v, numeric.f_to_v),
        cat(boolean.edge_attr, numeric.edge_attr),
        boolean.n_variable + numeric.n_variable,
        boolean.n_factor,
    )


class GraphAgent(nn.Module):
    def __init__(
        self,
        config: AgentConfig,
        rngs: Rngs,
        device: str = "cpu",
    ):
        super().__init__()  # type: ignore

        gnn_params = config.hyper_params

        self.config = config
        self.factor_embedding = EmbeddingLayer(
            config.num_object_classes, gnn_params.embedding_dim, rngs
        )

        self.predicate_embedding = EmbeddingLayer(
            config.num_predicate_classes, gnn_params.embedding_dim, rngs
        )

        self.edge_attr_embedding = EmbeddingLayer(
            config.arity, gnn_params.embedding_dim, rngs, use_padding=False
        )

        self.boolean_embedder = (
            NegativeBiasBooleanEmbedder(
                gnn_params.embedding_dim,
                self.predicate_embedding,
                rngs,
            )
            if config.remove_false_fluents
            else BooleanEmbedder(
                gnn_params.embedding_dim,
                self.predicate_embedding,
                rngs,
            )
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
        self.device = device

    def embed(self, data: HeteroBatchData) -> FactorGraph:
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

    def forward(self, actions: Tensor, data: HeteroBatchData):
        fg = self.embed(data)
        return self.policy(
            actions,
            fg.factors,
            data.boolean.action_type_mask,
            data.boolean.action_arity_mask,
            fg.n_factor,
        )

    def sample_from_obs(
        self,
        obs: Mapping[str, (list[ObsData] | tuple[ObsData, ...])],
        deterministic: bool = False,
    ):
        s = single_obs_to_heterostatedata(obs)
        s = heterostatedata_to_tensors(s, device=self.device)
        return self.sample(s, deterministic=deterministic)

    def sample(self, data: HeteroBatchData, deterministic: bool = False):
        fg = self.embed(data)
        return self.policy.sample(
            fg.factors,
            fg.n_factor,
            data.boolean.action_type_mask,
            data.boolean.action_arity_mask,
            deterministic,
        )

    def value(self, data: HeteroBatchData):
        fg = self.embed(data)
        return self.policy.value(
            fg.factors,
            fg.n_factor,
            data.boolean.action_type_mask,
            data.boolean.action_arity_mask,
        )

    def save_agent(self, path: str):
        save_agent(self, self.config, path)

    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def check_compatability(self, model: BaseModel):
        assert (
            self.config.num_object_classes == model.num_types
        ), "Mismatch in number of variable types, agent expects {}, model has {}".format(
            self.config.num_object_classes, model.num_types
        )
        assert (
            self.config.num_predicate_classes == model.num_fluents
        ), "Mismatch in number of predicates, agent expects {}, model has {}".format(
            self.config.num_predicate_classes, model.num_fluents
        )
        assert (
            self.config.num_actions == model.num_actions
        ), "Mismatch in number of action types, agent expects {}, model has {}".format(
            self.config.num_actions, model.num_actions
        )

    @classmethod
    def load_agent(
        cls, path: str, device: str = "cpu"
    ) -> tuple["GraphAgent", AgentConfig]:
        return load_agent(cls, path, device)  # type: ignore


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

    def embed(self, data: HeteroBatchData) -> FactorGraph:
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

    def forward(self, actions: Tensor, data: HeteroBatchData):
        fg = self.embed(data)
        return self.policy(actions, fg.factors, fg.action_mask, fg.n_factor)

    def sample(self, data: HeteroBatchData, deterministic: bool = False):
        fg = self.embed(data)
        return self.policy.sample(
            fg.factors, fg.n_factor, fg.action_mask, deterministic
        )

    def value(self, data: HeteroBatchData):
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


def load_agent(cls: T, path: str, device: str = "cpu") -> tuple[T, AgentConfig]:
    data = torch.load(path, weights_only=False, map_location=device)  # type: ignore

    data["config"]["hyper_params"] = GNNParams(**data["config"]["hyper_params"])

    if "remove_false_fluents" not in data["config"]:
        data["config"]["remove_false_fluents"] = False  # for backward compatibility

    config = AgentConfig(**data["config"])
    agent = cls(config, None)
    agent.load_state_dict(data["state_dict"])

    return agent, config
