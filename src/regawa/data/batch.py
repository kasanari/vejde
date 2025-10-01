import numpy as np
from numpy.typing import NDArray

from typing import Generic, NamedTuple

from .graph import VariableDomain

from .obs import ObsData
from .sparse import SparseArray


class BatchData(NamedTuple, Generic[VariableDomain]):
    """This represents a batch of multiple factor graphs."""

    var_value: SparseArray[VariableDomain]
    var_type: SparseArray[np.int64]
    factor: SparseArray[np.int64]
    v_to_f: NDArray[np.int64]  # variable
    f_to_v: NDArray[np.int64]  # factor
    edge_attr: NDArray[np.int64]
    n_factor: NDArray[np.int64]
    n_variable: NDArray[np.int64]
    n_graphs: np.int64
    length: NDArray[np.int64]
    global_vars: SparseArray[np.int64]
    global_vals: SparseArray[VariableDomain]
    global_length: NDArray[np.int64]
    action_arity_mask: NDArray[np.bool_]
    action_type_mask: NDArray[np.bool_]


class HeteroBatchData(NamedTuple):
    """This represents a batch of multiple heterogeneous factor graphs."""

    boolean: BatchData[np.int8]
    numeric: BatchData[np.float32]

    @property
    def n_graphs(self) -> np.int64:
        return self.boolean.n_graphs


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

    # Factors
    total_factors = sum(g.factor.n_factor for g in graphs)
    factor = np.empty((total_factors,), dtype=np.int64)
    factor_batch = np.empty((total_factors,), dtype=np.int64)

    # Edges
    total_edges = sum(g.edges.v_to_f.size for g in graphs)
    senders = np.empty((total_edges,), dtype=np.int64)
    receivers = np.empty((total_edges,), dtype=np.int64)
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
        var_value[variable_offsets : variable_offsets + flat_var_len] = g.var.value
        var_type[variable_offsets : variable_offsets + flat_var_len] = g.var.types
        var_batch[variable_offsets : variable_offsets + flat_var_len] = i
        length[num_vars_offset : num_vars_offset + num_vars] = g.var.length

        # Factors
        fac_len = g.factor.n_factor
        factor[factor_offsets : factor_offsets + fac_len] = g.factor.types
        factor_batch[factor_offsets : factor_offsets + fac_len] = i

        # Edges
        senders[edge_offsets : edge_offsets + edge_len] = (
            g.edges.v_to_f + num_vars_offset
        )  # don't offset vars by their full length, since the vars will be flattened before message passing
        receivers[edge_offsets : edge_offsets + edge_len] = (
            g.edges.f_to_v + factor_offsets
        )
        edge_attr[edge_offsets : edge_offsets + edge_len] = g.edges.edge_attr

        # Global Variables
        flat_globals_len = g.global_var.value.size
        num_globals_vars = g.global_var.length.shape[0]
        global_vars[globals_offset : globals_offset + flat_globals_len] = (
            g.global_var.types
        )
        global_vals[globals_offset : globals_offset + flat_globals_len] = (
            g.global_var.value
        )
        global_length[num_globals_offset : num_globals_offset + num_globals_vars] = (
            g.global_var.length
        )
        global_batch[globals_offset : globals_offset + flat_globals_len] = i

        # Action masks
        action_arity_mask[factor_offsets : factor_offsets + fac_len] = (
            g.action_masks.action_arity_mask
        )
        action_type_mask[factor_offsets : factor_offsets + fac_len] = (
            g.action_masks.action_type_mask
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
        var_value=SparseArray(var_value, var_batch),
        var_type=SparseArray(var_type, var_batch),
        factor=SparseArray(factor, factor_batch),
        edge_attr=edge_attr,
        v_to_f=senders,
        f_to_v=receivers,
        n_factor=n_factor,
        n_variable=n_variable,
        n_graphs=np.int64(num_graphs),
        length=length,
        global_vars=SparseArray(global_vars, global_batch),
        global_vals=SparseArray(global_vals, global_batch),
        global_length=global_length,
        action_arity_mask=action_arity_mask,
        action_type_mask=action_type_mask,
    )
