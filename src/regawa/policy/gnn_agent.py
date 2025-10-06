from __future__ import annotations
from pathlib import Path

import torch.nn as nn
from torch import Generator as Rngs
from torch import Tensor

from regawa.data import TorchFactorGraph, heterostatedata_to_tensors
from regawa.data import HeteroObsData

from regawa.embedding import (
    NegativeBiasBooleanEmbedder,
    NumericEmbedder,
)
from regawa.model import BaseModel
from regawa.policy.save import save_agent

from .node_then_action import NodeThenActionPolicy
from .action_then_node import ActionThenNodePolicy
from regawa.data import (
    HeteroBatchData,
    single_obs_to_heterostatedata,
)
from regawa.gnn import BipartiteGNN
from regawa.embedding import (
    EmbeddingLayer,
    fn_embed_graph,
    fn_embed_heterobatch,
)
from .agent_config import ActionMode, AgentConfig
from abc import ABC, abstractmethod


class GraphAgentInterface(ABC):
    @abstractmethod
    def __init__(self, config: AgentConfig, rngs: Rngs, device: str = "cpu"): ...

    @abstractmethod
    def embed(self, data: HeteroBatchData) -> TorchFactorGraph: ...

    @abstractmethod
    def forward(self, actions: Tensor, data: HeteroBatchData) -> tuple[Tensor, ...]: ...

    @abstractmethod
    def sample_from_obs(
        self,
        obs: HeteroObsData,
        deterministic: bool = False,
    ) -> tuple[Tensor, ...]: ...

    @abstractmethod
    def sample(
        self, data: HeteroBatchData, deterministic: bool = False
    ) -> tuple[Tensor, ...]: ...

    @abstractmethod
    def value(self, data: HeteroBatchData) -> Tensor: ...

    @abstractmethod
    def save_agent(self, path: str | Path): ...

    @abstractmethod
    def num_trainable_params(self) -> int: ...

    @abstractmethod
    def check_compatability(self, model: BaseModel): ...

    @property
    def device(self) -> str: ...

    @device.setter
    def device(self, device: str) -> None: ...


class GraphAgent(nn.Module, GraphAgentInterface):
    def __init__(
        self,
        config: AgentConfig,
        rngs: Rngs,
        device: str = "cpu",
    ):
        super().__init__()  # type: ignore

        gnn_params = config.hyper_params

        self.config = config
        factor_embedding = EmbeddingLayer(
            config.num_object_classes,
            gnn_params.embedding_dim,
            rngs,
        )

        predicate_embedding = EmbeddingLayer(
            config.num_predicate_classes,
            gnn_params.embedding_dim,
            rngs,
        )

        edge_attr_embedding = EmbeddingLayer(
            config.arity, gnn_params.embedding_dim, rngs, use_padding=False
        )

        boolean_embedder = NegativeBiasBooleanEmbedder(
            gnn_params.embedding_dim,
            predicate_embedding,
            rngs,
        )

        numeric_embedder = NumericEmbedder(
            gnn_params.embedding_dim,
            gnn_params.activation,
            predicate_embedding,
        )

        self.message_pass = BipartiteGNN(
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
        self._device = device
        self.boolean_embedder = boolean_embedder
        self.numeric_embedder = numeric_embedder
        self.factor_embedding = factor_embedding
        self.edge_attr_embedding = edge_attr_embedding
        self.predicate_embedding = predicate_embedding
        self.embed_heterobatch = fn_embed_heterobatch(
            fn_embed_graph(
                boolean_embedder,
                factor_embedding,
                boolean_embedder,
                edge_attr_embedding,
            ),
            fn_embed_graph(
                numeric_embedder,
                factor_embedding,
                numeric_embedder,
                edge_attr_embedding,
            ),
        )

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, device: str) -> None:
        self._device = device
        self.to(device)

    def embed(self, data: HeteroBatchData) -> TorchFactorGraph:
        return self.message_pass(self.embed_heterobatch(data))

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
        obs: HeteroObsData,
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

    def save_agent(self, path: str | Path):
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
