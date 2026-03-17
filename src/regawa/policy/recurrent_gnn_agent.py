from pathlib import Path
from typing import TypeVar

import numpy as np
from torch import Generator as Rngs
from torch import Tensor, nn

from regawa.data import (
    HeteroObsData,
    TorchFactorGraph,
    TorchHeteroBatchData,
    heterostatedata_to_tensors,
    single_obs_to_heterostatedata,
)
from regawa.embedding import (
    EmbeddingLayer,
    NumericEmbedder,
    PositiveNegativeBooleanEmbedder,
    RecurrentEmbedder,
    fn_compress_time,
    fn_embed_graph,
    fn_embed_heterobatch,
)
from regawa.gnn import BipartiteGNN
from regawa.model import BaseModel

from . import ActionMode, AgentConfig
from .action_then_node import ActionThenNodePolicy
from .graph_agent_interface import GraphAgentInterface
from .node_then_action import NodeThenActionPolicy
from .save import save_agent

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

        self._config = config
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
        embed_batch_func = fn_embed_heterobatch(
            fn_compress_time(
                r_boolean_embedder,
                fn_embed_graph(
                    boolean_embedder,
                    factor_embedding,
                    edge_attr_embedding,
                ),
                100,
                gnn_params.embedding_dim,
                device=device,
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
                device=device,
            ),
        )

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
        self.embed_heterobatch = embed_batch_func

    @property
    def device(self) -> str:
        return self._device

    @property
    def config(self) -> AgentConfig:
        return self._config

    @device.setter
    def device(self, device: str) -> None:
        self._device = device
        self.to(device)

    # Listening to: Sagittarius by Daisuke Achiwa
    def embed(self, data: TorchHeteroBatchData) -> TorchFactorGraph:
        return self.p_gnn(self.embed_heterobatch(data))

    def forward(self, actions: Tensor, data: TorchHeteroBatchData):
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

    def sample(self, data: TorchHeteroBatchData, deterministic: bool = False):
        fg = self.embed(data)
        return self.policy.sample(
            fg.factors,
            fg.n_factor,
            data.boolean.action_masks,
            deterministic,
        )

    def value(self, data: TorchHeteroBatchData):
        fg = self.embed(data)
        return self.policy.value(
            fg.factors,
            fg.n_factor,
            data.boolean.action_masks,
        )

    def save_agent(self, path: str | Path, model: BaseModel | None = None):
        save_agent(self, self.config, path, model=model)

    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def check_compatability(self, model: BaseModel):
        assert (
            self.config.num_object_classes == model.num_types
        ), f"Mismatch in number of variable types, agent expects {self.config.num_object_classes}, model has {model.num_types}"
        assert (
            self.config.num_predicate_classes == model.num_fluents
        ), f"Mismatch in number of predicates, agent expects {self.config.num_predicate_classes}, model has {model.num_fluents}"
        assert (
            self.config.num_actions == model.num_actions
        ), f"Mismatch in number of action types, agent expects {self.config.num_actions}, model has {model.num_actions}"
