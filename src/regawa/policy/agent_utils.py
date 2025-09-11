from dataclasses import dataclass
from enum import Enum

import torch.nn as nn
from typing import TypeVar

import numpy as np
import torch
from torch import Tensor, as_tensor, concatenate, int64

from regawa.data import FactorGraph, SparseTensor, sparsify


from regawa.data import (
    BatchData,
)


class ActionMode(Enum):
    ACTION_THEN_NODE = 0
    NODE_THEN_ACTION = 1
    ACTION_AND_NODE = 2


@dataclass
class GNNParams:
    embedding_dim: int
    layers: int
    aggregation: str
    activation: nn.Module
    action_mode: ActionMode
    recurrent = False


@dataclass
class AgentConfig:
    # environment parameters
    num_object_classes: int
    num_predicate_classes: int
    num_actions: int
    remove_false_fluents: bool

    # GNN parameters
    hyper_params: GNNParams
    arity: int


V = TypeVar("V", np.float32, np.bool_)


def embed(
    data: BatchData[V],
    var_embedder: nn.Module,
    factor_embedding: nn.Module,
    global_var_embedder: nn.Module,
    edge_attr_emb: nn.Module,
) -> FactorGraph:
    factors = sparsify(factor_embedding)(data.factor)
    variables = SparseTensor(
        var_embedder(
            data.var_value.values,
            data.var_type.values,
        ),
        data.var_value.indices,
    )
    globals_ = (
        SparseTensor(
            global_var_embedder(
                data.global_vals.values,
                data.global_vars.values,
            ),
            data.global_vals.indices,
        )
        if data.global_vals.shape[0] > 0
        else SparseTensor(
            as_tensor([], device=data.global_vals.values.device),
            as_tensor([], dtype=int64, device=data.global_vals.values.device),
        )
    )

    embedd_edge_attr = edge_attr_emb(data.edge_attr)

    return FactorGraph(
        variables,
        factors,
        globals_,
        data.v_to_f,
        data.f_to_v,
        embedd_edge_attr,
        data.n_variable,
        data.n_factor,
    )


def cat(a: Tensor, b: Tensor) -> Tensor:
    return concatenate((a, b))


@torch.jit.script
def concat_sparse(a: SparseTensor, b: SparseTensor) -> SparseTensor:
    return SparseTensor(
        concatenate((a.values, b.values)),
        concatenate((a.indices, b.indices)),
    )


@torch.jit.script
def merge_graphs(
    boolean: FactorGraph,
    numeric: FactorGraph,
) -> FactorGraph:
    # this only refers to the factors, so we can use either boolean or numeric data

    return FactorGraph(
        concat_sparse(boolean.variables, numeric.variables),
        # same factors for both boolean and numeric data, so we can use either
        boolean.factors,
        concat_sparse(boolean.globals, numeric.globals),
        cat(boolean.v_to_f, numeric.v_to_f + sum(boolean.n_variable)),
        # since factors are the same, we do not need to offset the receiver indices
        cat(boolean.f_to_v, numeric.f_to_v),
        cat(boolean.edge_attr, numeric.edge_attr),
        boolean.n_variable + numeric.n_variable,
        boolean.n_factor,
    )


def save_agent(agent: GraphAgent | RecurrentGraphAgent, config: AgentConfig, path: str):
    state_dict = agent.state_dict()
    to_save: dict[str, Any] = {}
    to_save["config"] = asdict(config)
    to_save["state_dict"] = state_dict
    torch.save(to_save, path)  # type: ignore


T = TypeVar("T", bound=GraphAgent | RecurrentGraphAgent)


def load_agent(cls: T, path: str, device: str = "cpu") -> tuple[T, AgentConfig]:
    data = torch.load(path, weights_only=False, map_location=device)  # type: ignore

    data["config"]["hyper_params"] = GNNParams(**data["config"]["hyper_params"])

    if "remove_false_fluents" not in data["config"]:
        data["config"]["remove_false_fluents"] = False  # for backward compatibility

    config = AgentConfig(**data["config"])
    agent = cls(config, None)
    agent.load_state_dict(data["state_dict"])

    return agent, config
