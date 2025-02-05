import torch.nn as nn


from regawa.gnn.agent_utils import ActionMode, AgentConfig
from regawa.gnn.gnn_agent import _embed, merge_graphs
from regawa.gnn.q_action_then_node import QActionThenNode
from regawa.gnn.q_node_then_action import QNodeThenAction

from .gnn_classes import EmbeddingLayer
from .data import HeteroStateData
from .factorgraph_gnn import BipartiteGNN, FactorGraph
from .gnn_embedder import (
    NegativeBiasBooleanEmbedder,
    NumericEmbedder,
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

    def embed(self, data: HeteroStateData) -> FactorGraph:
        return self.p_gnn(
            merge_graphs(
                _embed(
                    data.boolean,
                    self.boolean_embedder,
                    self.factor_embedding,
                    self.boolean_embedder,
                ),
                _embed(
                    data.numeric,
                    self.numeric_embedder,
                    self.factor_embedding,
                    self.numeric_embedder,
                ),
            )
        )

    def forward(self, data: HeteroStateData):
        return self.qfunc.forward(self.embed(data).factors)

    def save_agent(self, path: str):
        save_agent(self, self.config, path)

    @classmethod
    def load_agent(cls, path: str) -> tuple["RecurrentGraphAgent", AgentConfig]:
        return load_agent(cls, path)  # type: ignore
