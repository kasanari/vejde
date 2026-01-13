import numpy as np
from numpy.typing import NDArray

from typing import Generic, NamedTuple, TypeVar

from .actions import ActionMask

from .graph import VariableDomain, Edges

from .obs import ObsData
from .sparse import SparseArray


class BatchedVariables(NamedTuple, Generic[VariableDomain]):
    var_value: SparseArray[VariableDomain]
    var_type: SparseArray[np.int64]
    n_variable: NDArray[np.int64]
    length: NDArray[np.int64]
    times: NDArray[np.int64]


class BatchedFactors(NamedTuple):
    factor: SparseArray[np.int64]
    n_factor: NDArray[np.int64]


class BatchData(NamedTuple, Generic[VariableDomain]):
    """This represents a batch of multiple factor graphs."""

    factor: BatchedFactors
    variables: BatchedVariables[VariableDomain]
    edges: Edges
    n_graphs: np.int64
    global_variables: BatchedVariables[VariableDomain]
    action_masks: ActionMask


class HeteroBatchData(NamedTuple):
    """This represents a batch of multiple heterogeneous factor graphs."""

    boolean: BatchData[np.int8]
    numeric: BatchData[np.float32]

    @property
    def n_graphs(self) -> np.int64:
        return self.boolean.n_graphs

    @property
    def n_factor(self) -> NDArray[np.int64]:
        # The assumption at the moment is that both boolean and numeric use the same factors, even if one might have no variables.
        return self.boolean.factor.n_factor


ArrayDomain = TypeVar("ArrayDomain", np.int8, np.float32, np.bool_, np.int64)


def add_to_array(
    arr: NDArray[ArrayDomain],
    to_add: NDArray[ArrayDomain] | int,
    start: int,
    length: int,
) -> None:
    arr[start : start + length] = to_add


def batch(graphs: list[ObsData[VariableDomain]]) -> BatchData[VariableDomain]:
    """
    This is a big ugly function that batches multiple factor graphs into a single one.
    Its uglyness comes from a need for speed and memory efficiency in this particular function, as it is called many times during training.
    """

    # to get dtypes and shapes, which are assumed to be the same for all graphs
    g0 = graphs[0]

    # Variables
    total_length = sum(
        g.var.n_variable for g in graphs
    )  # 1 length per variable, even with stacking
    total_variables = sum(
        sum(g.var.length) for g in graphs
    )  # account for stacking. each variable can have length
    var_value = np.empty((total_variables,), dtype=g0.var.value.dtype)
    var_type = np.empty((total_variables,), dtype=np.int64)
    var_batch = np.empty((total_variables,), dtype=np.int64)
    length = np.empty((total_length,), dtype=np.int64)
    times = np.empty((total_variables, 2), dtype=np.int64)

    # Factors
    total_factors = sum(g.factor.n_factor for g in graphs)
    factor = np.empty((total_factors,), dtype=np.int64)
    factor_batch = np.empty((total_factors,), dtype=np.int64)

    # Edges
    total_edges = sum(g.edges.v_to_f.size for g in graphs)
    v_to_f = np.empty((total_edges,), dtype=np.int64)
    f_to_v = np.empty((total_edges,), dtype=np.int64)
    edge_attr = np.empty((total_edges,), dtype=np.int64)

    # Graph info
    num_graphs = len(graphs)
    n_factor = np.empty((num_graphs,), dtype=np.int64)
    n_variable = np.empty((num_graphs,), dtype=np.int64)

    # Global Variables
    flat_total_globals = sum(g.global_var.value.size for g in graphs)
    total_global_vars = sum(g.global_var.length.shape[0] for g in graphs)
    global_vars = np.empty((flat_total_globals,), dtype=np.int64)
    global_vals = np.empty((flat_total_globals,), dtype=g0.global_var.value.dtype)
    global_length = np.empty((total_global_vars,), dtype=np.int64)
    global_batch = np.empty((flat_total_globals,), dtype=np.int64)
    global_times = np.empty((flat_total_globals, 2), dtype=np.int64)

    # Action masks
    action_arity_mask = np.empty(
        (total_factors, g0.action_masks.action_arity_mask.shape[1]), dtype=np.bool_
    )
    action_type_mask = np.empty(
        (total_factors, g0.action_masks.action_type_mask.shape[1]), dtype=np.bool_
    )

    # Offsets, to keep track of where we are in the big arrays
    (
        factor_offsets,
        variable_offsets,
        globals_offset,
        num_vars_offset,
        num_globals_offset,
        edge_offsets,
    ) = 0, 0, 0, 0, 0, 0
    for i, g in enumerate(graphs):
        # Variables
        flat_var_len = sum(
            g.var.length
        )  # account for stacking. each variable can have length
        num_vars = g.var.n_variable  # 1 length per variable, even with stacking
        edge_len = g.edges.v_to_f.size
        add_to_array(var_value, g.var.value, variable_offsets, flat_var_len)
        add_to_array(var_type, g.var.types, variable_offsets, flat_var_len)
        add_to_array(var_batch, i, variable_offsets, flat_var_len)
        add_to_array(
            times, g.var.times, variable_offsets, flat_var_len
        ) if g.var.times.size > 0 else None
        add_to_array(length, g.var.length, num_vars_offset, num_vars)

        # Factors
        fac_len = g.factor.n_factor
        add_to_array(factor, g.factor.types, factor_offsets, fac_len)
        add_to_array(factor_batch, i, factor_offsets, fac_len)

        # Edges
        # don't offset vars by their full length, since the vars will be flattened before message passing
        add_to_array(v_to_f, g.edges.v_to_f + num_vars_offset, edge_offsets, edge_len)
        add_to_array(f_to_v, g.edges.f_to_v + factor_offsets, edge_offsets, edge_len)
        add_to_array(edge_attr, g.edges.edge_attr, edge_offsets, edge_len)

        # Global Variables
        flat_globals_len = g.global_var.value.size
        num_globals_vars = g.global_var.length.shape[0]
        add_to_array(global_vars, g.global_var.types, globals_offset, flat_globals_len)
        add_to_array(global_vals, g.global_var.value, globals_offset, flat_globals_len)
        add_to_array(
            global_length, g.global_var.length, num_globals_offset, num_globals_vars
        )
        add_to_array(global_batch, i, globals_offset, flat_globals_len)
        add_to_array(
            global_times, g.global_var.times, globals_offset, flat_globals_len
        ) if g.global_var.times.size > 0 else None

        # Action masks
        add_to_array(
            action_arity_mask, g.action_masks.action_arity_mask, factor_offsets, fac_len
        )
        add_to_array(
            action_type_mask, g.action_masks.action_type_mask, factor_offsets, fac_len
        )

        # Graph info
        n_factor[i] = fac_len
        n_variable[i] = num_vars

        # Update offsets
        factor_offsets += fac_len
        variable_offsets += flat_var_len
        num_vars_offset += num_vars
        edge_offsets += edge_len
        globals_offset += flat_globals_len
        num_globals_offset += num_globals_vars

    return BatchData(
        variables=BatchedVariables(
            var_value=SparseArray(var_value, var_batch),
            var_type=SparseArray(var_type, var_batch),
            n_variable=n_variable,
            length=length,
            times=times,
        ),
        edges=Edges(v_to_f=v_to_f, f_to_v=f_to_v, edge_attr=edge_attr),
        factor=BatchedFactors(
            factor=SparseArray(factor, factor_batch),
            n_factor=n_factor,
        ),
        n_graphs=np.int64(num_graphs),
        global_variables=BatchedVariables(
            var_value=SparseArray(global_vals, global_batch),
            var_type=SparseArray(global_vars, global_batch),
            n_variable=global_length,
            length=global_length,
            times=global_times,
        ),
        action_masks=ActionMask(
            action_arity_mask=action_arity_mask, action_type_mask=action_type_mask
        ),
    )
