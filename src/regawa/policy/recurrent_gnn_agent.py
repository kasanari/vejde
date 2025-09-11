from typing import TypeVar

import numpy as np
import torch.nn as nn
from torch import Generator as Rngs
from torch import Tensor

from regawa.data import FactorGraph
from regawa.policy.save import save_agent

from . import ActionMode, AgentConfig

from .node_then_action import NodeThenActionPolicy
from .action_then_node import ActionThenNodePolicy
from regawa.data import (
    HeteroBatchData,
)
from regawa.gnn import BipartiteGNN
from regawa.embedding import (
    BooleanEmbedder,
    NumericEmbedder,
    RecurrentEmbedder,
    EmbeddingLayer,
)
from .agent_utils import embed, merge_graphs

V = TypeVar("V", np.float32, np.bool_)


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
            rngs,
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
            rngs,
        )

        policy_args = (config.num_actions, gnn_params.embedding_dim, rngs)
        self.policy = (
            ActionThenNodePolicy(*policy_args)
            if gnn_params.action_mode == ActionMode.ACTION_THEN_NODE
            else NodeThenActionPolicy(*policy_args)
        )

    def embed(self, data: HeteroBatchData) -> FactorGraph:
        return self.p_gnn(
            merge_graphs(
                embed(
                    data.boolean,
                    self.r_boolean_embedder(data.boolean.length),
                    self.factor_embedding,
                    self.r_boolean_embedder(data.boolean.global_length),
                    self.edge_attr_embedding,
                ),
                embed(
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
        return self.policy(
            actions,
            fg.factors,
            data.boolean.action_type_mask,
            data.boolean.action_arity_mask,
            fg.n_factor,
        )

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
        _, _, _, value, *_ = self.policy.sample(
            fg.factors,
            fg.n_factor,
            data.boolean.action_type_mask,
            data.boolean.action_arity_mask,
            False,
        )
        return value

    def save_agent(self, path: str):
        save_agent(self, self.config, path)
