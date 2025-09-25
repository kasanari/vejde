import torch.nn as nn

from regawa.embedding.boolean import NegativeBiasBooleanEmbedder
from regawa.embedding.numeric import NumericEmbedder
from regawa.policy import ActionMode, AgentConfig
from .q_action_then_node import QActionThenNode
from .q_node_then_action import QNodeThenAction

from regawa.data import TorchFactorGraph, HeteroBatchData
from regawa.gnn import BipartiteGNN
from regawa.embedding import (
    EmbeddingLayer,
    merge_graphs,
)


class GraphQAgent(nn.Module):
    def __init__(
        self,
        config: AgentConfig,
    ):
        super().__init__()  # type: ignore

        self.config = config

        self.factor_embedding = EmbeddingLayer(
            config.num_object_classes, config.embedding_dim
        )

        self.predicate_embedding = EmbeddingLayer(
            config.num_predicate_classes, config.embedding_dim
        )

        self.boolean_embedder = NegativeBiasBooleanEmbedder(
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

        qfunc = (
            QActionThenNode
            if config.action_mode == ActionMode.ACTION_THEN_NODE
            else QNodeThenAction
        )
        self.qfunc = qfunc(config.num_actions, config.embedding_dim)

    def embed(self, data: HeteroBatchData) -> TorchFactorGraph:
        return self.p_gnn(
            merge_graphs(
                embed(
                    data.boolean,
                    self.boolean_embedder,
                    self.factor_embedding,
                    self.boolean_embedder,
                ),
                embed(
                    data.numeric,
                    self.numeric_embedder,
                    self.factor_embedding,
                    self.numeric_embedder,
                ),
            )
        )

    def forward(self, data: HeteroBatchData):
        return self.qfunc.forward(self.embed(data).factors)
