from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

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
    NegativeBiasBooleanEmbedder,
    NumericEmbedder,
    fn_embed_graph,
    fn_embed_heterobatch,
)
from regawa.gnn import BipartiteGNN
from regawa.model import BaseModel

from .action_then_node import ActionThenNodePolicy
from .agent_config import ActionMode, AgentConfig
from .node_then_action import NodeThenActionPolicy
from .save import save_agent
from .types import PolicyOutput


class GraphAgentInterface(ABC):
    @abstractmethod
    def __init__(self, config: AgentConfig, rngs: Rngs, device: str = "cpu"): ...

    @abstractmethod
    def embed(self, data: TorchHeteroBatchData) -> TorchFactorGraph: ...

    @abstractmethod
    def forward(self, actions: Tensor, data: TorchHeteroBatchData) -> PolicyOutput: ...

    @abstractmethod
    def sample_from_obs(
        self,
        obs: HeteroObsData,
        deterministic: bool = False,
    ) -> PolicyOutput: ...

    @abstractmethod
    def sample(
        self, data: TorchHeteroBatchData, deterministic: bool = False
    ) -> PolicyOutput: ...

    @abstractmethod
    def value(self, data: TorchHeteroBatchData) -> Tensor: ...

    @abstractmethod
    def save_agent(self, path: str | Path): ...

    @abstractmethod
    def num_trainable_params(self) -> int: ...

    @abstractmethod
    def check_compatability(self, model: BaseModel): ...

    @property
    @abstractmethod
    def device(self) -> str: ...

    @device.setter
    @abstractmethod
    def device(self, device: str) -> None: ...

    @property
    @abstractmethod
    def config(self) -> AgentConfig: ...


class GraphAgent(nn.Module, GraphAgentInterface):
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
                edge_attr_embedding,
            ),
            fn_embed_graph(
                numeric_embedder,
                factor_embedding,
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

    def embed(self, data: TorchHeteroBatchData) -> TorchFactorGraph:
        return self.message_pass(self.embed_heterobatch(data))

    @property
    def config(self) -> AgentConfig:
        return self._config

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
