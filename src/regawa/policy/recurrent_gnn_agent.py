from pathlib import Path
from typing import TypeVar

import numpy as np
import torch.nn as nn
from torch import Generator as Rngs
from torch import Tensor

from regawa.data import TorchFactorGraph
from regawa.data import single_obs_to_heterostatedata
from regawa.data import heterostatedata_to_tensors
from regawa.data.data import HeteroObsData
from regawa.embedding import (
    PositiveNegativeBooleanEmbedder,
    NumericEmbedder,
    RecurrentEmbedder,
)
from regawa.model.base_model import BaseModel
from .gnn_agent import GraphAgentInterface
from regawa.policy.save import save_agent

from . import ActionMode, AgentConfig

from .node_then_action import NodeThenActionPolicy
from .action_then_node import ActionThenNodePolicy
from regawa.data import (
    HeteroBatchData,
)
from regawa.gnn import BipartiteGNN
from regawa.embedding import EmbeddingLayer, fn_compress_time, fn_embed_heterobatch
from regawa.embedding import fn_embed_graph

V = TypeVar("V", np.float32, np.bool_)


class RecurrentGraphAgent(nn.Module, GraphAgentInterface):
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

        boolean_embedder = PositiveNegativeBooleanEmbedder(
            gnn_params.embedding_dim,
            predicate_embedding,
            rngs,
        )

        numeric_embedder = NumericEmbedder(
            gnn_params.embedding_dim,
            gnn_params.activation,
            predicate_embedding,
        )

        r_numeric_embedder = RecurrentEmbedder(
            gnn_params.embedding_dim,
            device=device,
        )

        r_boolean_embedder = RecurrentEmbedder(
            gnn_params.embedding_dim,
            device=device,
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
        self._device = device
        self.boolean_embedder = boolean_embedder
        self.numeric_embedder = numeric_embedder
        self.factor_embedding = factor_embedding
        self.edge_attr_embedding = edge_attr_embedding
        self.predicate_embedding = predicate_embedding
        self.r_numeric_embedder = r_numeric_embedder
        self.r_boolean_embedder = r_boolean_embedder

        self.embed_heterobatch = fn_embed_heterobatch(
            fn_compress_time(
                r_boolean_embedder,
                fn_embed_graph(
                    boolean_embedder,
                    factor_embedding,
                    edge_attr_embedding,
                ),
                100,
                gnn_params.embedding_dim,
            ),
            fn_compress_time(
                r_numeric_embedder,
                fn_embed_graph(
                    numeric_embedder,
                    factor_embedding,
                    edge_attr_embedding,
                ),
                100,
                gnn_params.embedding_dim,
            ),
        )

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, device: str) -> None:
        self._device = device
        self.to(device)

    # Listening to: Sagittarius by Daisuke Achiwa
    def embed(self, data: HeteroBatchData) -> TorchFactorGraph:
        return self.p_gnn(self.embed_heterobatch(data))

    def forward(self, actions: Tensor, data: HeteroBatchData):
        fg = self.embed(data)
        return self.policy(
            actions,
            fg.factors,
            data.boolean.action_masks,
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
            data.boolean.action_masks,
            deterministic,
        )

    def value(self, data: HeteroBatchData):
        fg = self.embed(data)
        return self.policy.value(
            fg.factors,
            fg.n_factor,
            data.boolean.action_masks,
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
