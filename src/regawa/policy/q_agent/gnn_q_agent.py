# type: ignore
import torch.nn as nn

from regawa.embedding.boolean import PositiveNegativeBooleanEmbedder
from regawa.embedding.numeric import NumericEmbedder
from regawa.policy import ActionMode, AgentConfig
from .q_action_then_node import QActionThenNode
from .q_node_then_action import QNodeThenAction

from regawa.data import TorchFactorGraph, HeteroBatchData
from regawa.gnn import BipartiteGNN
from regawa.embedding import (
    EmbeddingLayer,
)
from torch import Generator as Rngs
from regawa.embedding import (
    fn_embed_graph,
    fn_embed_heterobatch,
)

class GraphQAgent(nn.Module):
    def __init__(
        self,
        config: AgentConfig,
        rngs: Rngs,
        device: str = "cpu",
    ):
        super().__init__()  # type: ignore

        hyper_params = config.hyper_params
        embed_dim = config.hyper_params.embedding_dim
        factor_embedding = EmbeddingLayer(
            config.num_object_classes, embed_dim, rngs
        )

        predicate_embedding = EmbeddingLayer(
            config.num_predicate_classes, embed_dim, rngs
        )

        boolean_embedder = PositiveNegativeBooleanEmbedder(
            embed_dim,
            predicate_embedding,
            rngs,
        )

        numeric_embedder = NumericEmbedder(
            embed_dim,
            hyper_params.activation,
            predicate_embedding,
        )

        edge_attr_embedding = EmbeddingLayer(
            config.arity, embed_dim, rngs, use_padding=False
        )

        self.message_pass = BipartiteGNN(
            hyper_params.layers,
            embed_dim,
            hyper_params.aggregation,
            hyper_params.activation,
            rngs,
        )
        self.embed_heterobatch = fn_embed_heterobatch(
            fn_embed_graph(
                boolean_embedder,
                factor_embedding,
                edge_attr_embedding,
            ),
            fn_embed_graph(
                numeric_embedder,
                factor_embedding,
                edge_attr_embedding,
            ),
        )
        self.predicate_embedding = predicate_embedding
        self.factor_embedding = factor_embedding
        self.numeric_embedder = numeric_embedder
        self.edge_attr_embedding = edge_attr_embedding
        self.boolean_embedder = boolean_embedder
        self.config = config
        self._device = device
        self.to(device)

        qfunc = (
            QActionThenNode
            if hyper_params.action_mode == ActionMode.ACTION_THEN_NODE
            else QNodeThenAction
        )
        self.qfunc = qfunc(config.num_actions, embed_dim)


    def embed(self, data: HeteroBatchData) -> TorchFactorGraph:
        return self.message_pass(self.embed_heterobatch(data))

    def forward(self, data: HeteroBatchData):
        return self.qfunc.forward(self.embed(data).factors)
