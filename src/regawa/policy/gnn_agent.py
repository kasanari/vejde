from __future__ import annotations

import torch.nn as nn
from torch import Generator as Rngs
from torch import Tensor

from regawa.data import FactorGraph, heterostatedata_to_tensors
from regawa.data import HeteroObsData

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
    BooleanEmbedder,
    NegativeBiasBooleanEmbedder,
    NumericEmbedder,
    EmbeddingLayer,
)
from .agent_utils import ActionMode, AgentConfig, embed, merge_graphs


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
                embed(
                    data.boolean,
                    self.boolean_embedder,
                    self.factor_embedding,
                    self.boolean_embedder,
                    self.edge_attr_embedding,
                ),
                embed(
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
