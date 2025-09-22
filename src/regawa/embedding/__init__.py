from collections.abc import Callable
import torch.nn as nn
from torch import Tensor, as_tensor, concatenate, int64
from typing import TypeVar
import numpy as np
from regawa.data import BatchData, FactorGraph, SparseTensor, sparsify, HeteroBatchData
from regawa.data.torch import concat_sparse
from .recurrent import RecurrentEmbedder
from .boolean import (
    BooleanEmbedder,
    NegativeBiasBooleanEmbedder,
    PositiveNegativeBooleanEmbedder,
)
from .numeric import NumericEmbedder
from numpy.typing import NDArray
from .node_embedders import (
    EmbeddingLayer,
)

__all__ = [
    "BooleanEmbedder",
    "NegativeBiasBooleanEmbedder",
    "NumericEmbedder",
    "RecurrentEmbedder",
    "EmbeddingLayer",
    "PositiveNegativeBooleanEmbedder",
]


V = TypeVar("V", np.float32, np.bool_)


def cat(a: Tensor, b: Tensor) -> Tensor:
    return concatenate((a, b))


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
        cat(boolean.v_to_f, numeric.v_to_f + boolean.variables.values.shape[0]),
        # since factors are the same, we do not need to offset the receiver indices
        cat(boolean.f_to_v, numeric.f_to_v),
        cat(boolean.edge_attr, numeric.edge_attr),
        boolean.n_variable + numeric.n_variable,
        boolean.n_factor,
    )


def fn_embed_heterobatch(
    boolean_embedder: Callable[[BatchData[np.bool]], FactorGraph],
    numeric_embedder: Callable[[BatchData[np.float32]], FactorGraph],
):
    def embed_heterobatch(data: HeteroBatchData) -> FactorGraph:
        return merge_graphs(
            boolean_embedder(
                data.boolean,
            ),
            numeric_embedder(
                data.numeric,
            ),
        )

    return embed_heterobatch


def fn_embed_variables(
    var_embedder: Callable[[Tensor, Tensor], Tensor],
):
    def embed_variables(
        var_values: SparseTensor, var_types: SparseTensor
    ) -> SparseTensor:
        return (
            SparseTensor(
                var_embedder(
                    var_values.values,
                    var_types.values,
                ),
                var_values.indices,
            )
            if var_values.shape[0] > 0
            else SparseTensor(
                as_tensor([], device=var_values.values.device),
                as_tensor([], dtype=int64, device=var_values.values.device),
            )
        )

    return embed_variables


def fn_embed_graph(
    var_embedder: Callable[[Tensor, Tensor], Tensor],
    factor_embedding: Callable[[Tensor], Tensor],
    global_var_embedder: Callable[[Tensor, Tensor], Tensor],
    edge_attr_emb: nn.Module,
):
    var_embed = fn_embed_variables(var_embedder)
    global_var_embed = fn_embed_variables(global_var_embedder)
    factor_embed = sparsify(factor_embedding)

    def embed_graph(data: BatchData[V]) -> FactorGraph:
        return FactorGraph(
            var_embed(data.var_value, data.var_type),
            factor_embed(data.factor),
            global_var_embed(data.global_vals, data.global_vars),
            data.v_to_f,
            data.f_to_v,
            edge_attr_emb(data.edge_attr),
            data.n_variable,
            data.n_factor,
        )

    return embed_graph


def fn_compress_time(
    recurrent: Callable[[Tensor, NDArray[np.int64]], Tensor],
    embed_fn: Callable[[BatchData[V]], FactorGraph],
):
    def compress_time(data: BatchData[V]) -> FactorGraph:
        g = embed_fn(data)
        return g._replace(
            variables=SparseTensor(
                recurrent(g.variables.values, data.length), g.variables.indices
            ),
            globals=SparseTensor(
                recurrent(g.globals.values, data.global_length), g.globals.indices
            ),
        )

    return compress_time
