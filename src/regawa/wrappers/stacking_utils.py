from collections.abc import Callable, Sequence
from itertools import chain
from typing import TypeVar

import numpy as np

from .types import IdxFactorGraph, StackedFactorGraph, Variables
from .utils import map_graph_to_idx

V = TypeVar("V", np.float32, np.bool_)


def flatten(vals: Sequence[Sequence[V]], vars: Sequence[str]) -> Variables[V]:
    # Flatten the list of node history lists to account for different node history lengths
    flat_vals = list(chain(*vals))
    v = [[vars[i] for _ in v] for i, v in enumerate(vals)]  # expand the variable names
    flat_vars = list(chain(*v))
    lengths = [len(v) for v in vals]  # lengths of each variable history
    return Variables(flat_vars, flat_vals, lengths)


def flatten_values(
    factorgraph: StackedFactorGraph[V],
) -> tuple[Variables[V], Variables[V]]:
    return (
        flatten(factorgraph.variable_values, factorgraph.variables),
        flatten(factorgraph.global_variable_values, factorgraph.global_variables),
    )


def fn_flatten_map_graph_to_idx(
    rel_to_idx: Callable[[str], int], type_to_idx: Callable[[str], int]
):
    def flatten_map_graph_to_idx(
        factorgraph: StackedFactorGraph[V],
        var_val_dtype: type,
    ) -> IdxFactorGraph[V]:
        return map_graph_to_idx(
            *flatten_values(factorgraph),
            factorgraph.senders,
            factorgraph.receivers,
            factorgraph.edge_attributes,
            factorgraph.action_type_mask,
            factorgraph.action_arity_mask,
            factorgraph.factor_types,
            rel_to_idx,
            type_to_idx,
            var_val_dtype,
        )

    return flatten_map_graph_to_idx
