from collections.abc import Callable
from functools import partial
from typing import TypeVar

import numpy as np
import torch
from numpy.typing import NDArray
from torch import (
    FloatTensor,
    IntTensor,
    LongTensor,
    Tensor,
    as_tensor,
    concatenate,
    int64,
    nn,
)

from regawa.data import (
    SparseTensor,
    TorchBatchData,
    TorchFactorGraph,
    TorchHeteroBatchData,
    concat_sparse,
    sparsify,
)
from regawa.embedding.positional import pos_emb, rotate

from .boolean import (
    BooleanEmbeddingBooleanEmbedder,
    NegativeBiasBooleanEmbedder,
    PositiveNegativeBooleanEmbedder,
)
from .node_embedders import (
    EmbeddingLayer,
)
from .numeric import NumericEmbedder
from .recurrent import RecurrentEmbedder

__all__ = [
    "BooleanEmbeddingBooleanEmbedder",
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
    boolean: TorchFactorGraph,
    numeric: TorchFactorGraph,
) -> TorchFactorGraph:
    # this only refers to the factors, so we can use either boolean or numeric data

    return TorchFactorGraph(
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
    boolean_embedder: Callable[[TorchBatchData[IntTensor]], TorchFactorGraph],
    numeric_embedder: Callable[[TorchBatchData[FloatTensor]], TorchFactorGraph],
):
    def embed_heterobatch(data: TorchHeteroBatchData) -> TorchFactorGraph:
        return merge_graphs(
            boolean_embedder(
                data.boolean,
            ),
            numeric_embedder(
                data.numeric,
            ),
        )

    return embed_heterobatch


def fn_embed_variables[V: FloatTensor | IntTensor](
    var_embedder: Callable[[V, LongTensor], FloatTensor],
):
    def embed_variables(
        var_values: SparseTensor[V], var_types: SparseTensor[V]
    ) -> SparseTensor[FloatTensor]:
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
    var_embedder: Callable[[Tensor, LongTensor], FloatTensor],
    factor_embedding: Callable[[LongTensor], FloatTensor],
    edge_attr_emb: nn.Module,
):
    var_embed = fn_embed_variables(var_embedder)
    factor_embed = sparsify(factor_embedding)

    def embed_graph[V: FloatTensor | IntTensor](
        data: TorchBatchData[V],
    ) -> TorchFactorGraph:
        return TorchFactorGraph(
            var_embed(data.variables.var_value, data.variables.var_type),
            factor_embed(data.factor.factor),
            var_embed(data.global_variables.var_value, data.global_variables.var_type),
            data.edges.v_to_f,
            data.edges.f_to_v,
            edge_attr_emb(data.edges.edge_attr),
            data.variables.n_variable,
            data.factor.n_factor,
        )

    return embed_graph


def fn_compress_time(
    recurrent: Callable[
        [SparseTensor[FloatTensor], NDArray[np.int64]], SparseTensor[FloatTensor]
    ],
    embed_fn: Callable[[TorchBatchData[V]], TorchFactorGraph],
    maxlen: int,
    dim: int,
    k: float = 1e2,
    device: str | torch.device = "cpu",
):
    sin_freqs, cos_freqs = pos_emb(maxlen, dim, k)
    sin_freqs = as_tensor(sin_freqs, device=device, dtype=torch.float32)
    cos_freqs = as_tensor(cos_freqs, device=device, dtype=torch.float32)

    rot = partial(rotate, sin_freqs=sin_freqs, cos_freqs=cos_freqs)

    def compress_time(data: TorchBatchData[V]) -> TorchFactorGraph:
        g = embed_fn(data)

        start_times = data.variables.times[:, 0]
        # end_times = data.times[:, 1]
        durations = data.variables.times[:, 1] - data.variables.times[:, 0]
        global_start_times = data.global_variables.times[:, 0]
        # global_end_times = data.global_times[:, 1]
        global_durations = (
            data.global_variables.times[:, 1] - data.global_variables.times[:, 0]
        )

        variables_values = (
            g.variables.replace_val(
                rot(
                    rot(
                        g.variables.values,
                        as_tensor(start_times),
                    ),
                    as_tensor(durations),
                )
            )
            if g.variables.values.shape[0] > 0
            else g.variables
        )

        globals_values = (
            g.globals.replace_val(
                rot(
                    rot(
                        g.globals.values,
                        as_tensor(global_start_times),
                    ),
                    as_tensor(global_durations),
                )
            )
            if g.globals.values.shape[0] > 0
            else g.globals
        )

        return g._replace(
            variables=recurrent(variables_values, data.variables.length)
            if g.variables.values.shape[0] > 0
            else g.variables,
            globals=recurrent(globals_values, data.global_variables.length)
            if g.globals.values.shape[0] > 0
            else g.globals,
        )

    return compress_time
