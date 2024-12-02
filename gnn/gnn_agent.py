from dataclasses import dataclass
from typing import Any, TypeVar

import torch.nn as nn
from torch import Tensor

import torch as th
from dataclasses import asdict
from .data import StackedStateData, StateData
from .factorgraph_gnn import BipartiteGNN, FactorGraph
from .gnn_embedder import Embedder, RecurrentEmbedder
from .gnn_policies import ActionMode, TwoActionGNNPolicy


@dataclass
class Config:
    num_object_classes: int
    num_predicate_classes: int
    num_actions: int
    embedding_dim: int
    layers: int
    aggregation: str
    activation: nn.Module
    action_mode: ActionMode
    recurrent = False


class GraphAgent(nn.Module):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()  # type: ignore

        self.config = config

        self.embedder = Embedder(
            config.num_object_classes,
            config.num_predicate_classes,
            config.embedding_dim,
            config.activation,
        )

        self.p_gnn = BipartiteGNN(
            config.layers,
            config.embedding_dim,
            config.aggregation,
            config.activation,
        )

        self.actorcritic = GraphActorCritic(
            config.num_actions,
            config.embedding_dim,
            config.action_mode,
        )

    def embed(self, data: StateData) -> tuple[FactorGraph, Tensor]:
        variables, factors = self.embedder(
            data.var_val, data.var_type, data.object_class
        )
        fg = FactorGraph(
            variables,
            factors,
            data.edge_index,
            data.edge_attr,
            data.batch_idx,
        )
        e_fg = self.p_gnn(fg)
        return e_fg

    def forward(self, actions: Tensor, data: StateData):
        fg, g = self.embed(data)
        logprob, entropy, value = self.actorcritic(
            actions, fg.factors, g, data.batch_idx
        )
        return logprob, entropy, value

    def sample(self, data: StateData, deterministic: bool = False):
        fg, g = self.embed(data)
        action, logprob, entropy = self.actorcritic.sample(
            fg.factors, g, data.batch_idx, deterministic
        )
        value = self.actorcritic.value(g)
        return action, logprob, entropy, value

    def value(self, data: StateData):
        _, g = self.embed(data)
        return self.actorcritic.value(g)

    def save_agent(self, path: str):
        save_agent(self, self.config, path)

    @classmethod
    def load_agent(cls, path: str) -> tuple["RecurrentGraphAgent", Config]:
        return load_agent(cls, path)  # type: ignore


class RecurrentGraphAgent(nn.Module):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()  # type: ignore

        self.config = config

        self.embedder = RecurrentEmbedder(
            config.num_object_classes,
            config.num_predicate_classes,
            config.embedding_dim,
            config.activation,
        )

        self.p_gnn = BipartiteGNN(
            config.layers,
            config.embedding_dim,
            config.aggregation,
            config.activation,
        )

        self.actorcritic = GraphActorCritic(
            config.num_actions,
            config.embedding_dim,
            config.action_mode,
        )

    def embed(self, data: StackedStateData) -> tuple[FactorGraph, Tensor]:
        variables, factors = self.embedder(
            data.var_val, data.var_type, data.object_class, data.lengths
        )
        fg, g = self.p_gnn(
            FactorGraph(
                variables,
                factors,
                data.edge_index,
                data.edge_attr,
                data.batch_idx,
            )
        )
        return fg, g

    def forward(self, actions: Tensor, data: StackedStateData):
        fg, g = self.embed(data)
        o = (fg.factors, g)
        return self.actorcritic(actions, *o, data.batch_idx)

    def sample(self, data: StackedStateData, deterministic: bool = False):
        fg, g = self.embed(data)
        o = (fg.factors, g)
        action, logprob, entropy = self.actorcritic.sample(
            *o, data.batch_idx, deterministic
        )
        value = self.actorcritic.value(g)
        return action, logprob, entropy, value

    def value(self, data: StackedStateData):
        _, g = self.embed(data)
        return self.actorcritic.value(g)

    def save_agent(self, path: str):
        save_agent(self, self.config, path)

    @classmethod
    def load_agent(cls, path: str) -> tuple["RecurrentGraphAgent", Config]:
        return load_agent(cls, path)  # type: ignore


class GraphActorCritic(nn.Module):
    def __init__(
        self,
        num_actions: int,
        embedding_dim: int,
        action_mode: ActionMode,
    ):
        super().__init__()  # type: ignore
        self.policy = TwoActionGNNPolicy(num_actions, embedding_dim, action_mode)
        self.vf = nn.Linear(embedding_dim, 1)

    def value(
        self,
        g: Tensor,
    ):
        return self.vf(g)

    def forward(
        self,
        actions: Tensor,
        h: Tensor,
        g: Tensor,
        batch_idx: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return (
            *self.policy.forward(actions, h, g, batch_idx),
            self.vf(g),
        )

    def sample(
        self,
        h: Tensor,
        g: Tensor,
        batch_idx: Tensor,
        deterministic: bool = False,
    ):
        return self.policy.sample(h, g, batch_idx, deterministic)


def save_agent(agent: GraphAgent | RecurrentGraphAgent, config: Config, path: str):
    state_dict = agent.state_dict()
    to_save: dict[str, Any] = {}
    to_save["config"] = asdict(config)
    to_save["state_dict"] = state_dict
    th.save(to_save, path)  # type: ignore


T = TypeVar("T", bound=GraphAgent | RecurrentGraphAgent)


def load_agent(cls: T, path: str) -> tuple[T, Config]:
    data = th.load(path, weights_only=False)  # type: ignore
    config = Config(**data["config"])
    agent = cls(config)
    agent.load_state_dict(
        {k.replace("agent.", ""): v for k, v in data["state_dict"].items()}
    )

    return agent, config
