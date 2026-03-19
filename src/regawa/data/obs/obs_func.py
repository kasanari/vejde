from collections.abc import Callable

import numpy as np

from regawa.data.factor_graph.factor_graph import StringFactorGraph
from regawa.data.graph import (
    StringFactors,
    StringVariables,
    VariableDomain,
    Variables,
)
from regawa.data.heterograph import HeteroGraph
from regawa.data.stacked.stacked_graph_func import fn_flatten_then_map_graph_to_idx
from regawa.model import BaseModel

from .obs import Factors, GraphTypes, HeteroIndexedFactorGraph, IndexedFactorGraph


def fn_graph_to_obsdata(
    variables_to_idx: Callable[
        [StringVariables[VariableDomain], type], Variables[VariableDomain]
    ],
    factor_to_idx: Callable[[StringFactors], Factors],
):
    def map_graph_to_idx(
        g: StringFactorGraph[VariableDomain],
        var_val_dtype: type,
    ) -> IndexedFactorGraph[VariableDomain]:
        return IndexedFactorGraph(
            var=variables_to_idx(g.variables, var_val_dtype),
            factor=factor_to_idx(g.factors),
            edges=g.edges,
            global_var=variables_to_idx(g.global_variables, var_val_dtype),
            action_masks=g.action_masks,
        )

    return map_graph_to_idx


def fn_heterograph_to_heteroobs(
    fn_graph_to_idx: Callable[
        [
            GraphTypes,
            VariableDomain,
        ],
        IndexedFactorGraph[VariableDomain],
    ],
):
    """
    Returns a function that takes a HeteroGraph and returns a HeteroIndexedFactorGraph (for use in GNNs).
    """

    def heterograph_to_heteroobs(
        heterogenous_graph: HeteroGraph,
    ) -> HeteroIndexedFactorGraph:
        return HeteroIndexedFactorGraph(
            bool=fn_graph_to_idx(
                heterogenous_graph.boolean,
                np.int8,
            ),
            float=fn_graph_to_idx(
                heterogenous_graph.numeric,
                np.float32,
            ),
        )

    return heterograph_to_heteroobs


def factor_to_idx(type_to_idx: Callable[[str], int]):
    def map_factors_to_idx(
        factors: StringFactors,
    ) -> Factors:
        arr = np.asarray
        factor_type_idx = arr(
            [type_to_idx(f_type) for f_type in factors.types], dtype=np.int64
        )
        return Factors(
            factor_type_idx,
            factor_type_idx.shape[0],  # number of factors
        )

    return map_factors_to_idx


def fn_variables_to_idx(
    rel_to_idx: Callable[[str], int],
):
    def map_variables_to_idx(
        variables: StringVariables[VariableDomain], var_val_dtype: type
    ) -> Variables[VariableDomain]:
        arr = np.asarray
        return Variables(
            arr([rel_to_idx(p) for p in variables.types], dtype=np.int64),
            arr(variables.values, dtype=var_val_dtype),
            arr(variables.length),
            n_variable=variables.n_variable,
            times=np.zeros((variables.n_variable, 2), dtype=np.int64),
        )

    return map_variables_to_idx


def fn_variables_to_idx_with_time(
    rel_to_idx: Callable[[str], int],
):
    def map_variables_to_idx(
        variables: StringVariables[VariableDomain], var_val_dtype: type
    ) -> Variables[VariableDomain]:
        arr = np.asarray

        times, variable_values = (  # type: ignore
            zip(*variables.values, strict=False) if variables.values else ([], [])
        )

        return Variables(
            arr([rel_to_idx(p) for p in variables.types], dtype=np.int64),
            arr(variable_values, dtype=var_val_dtype),
            arr(variables.length),
            n_variable=variables.n_variable,
            times=arr(times, dtype=np.int64),
        )

    return map_variables_to_idx


def fn_idx_obs(model: BaseModel, stacking: bool = False):
    f = fn_graph_to_obsdata(
        fn_variables_to_idx_with_time(model.fluent_to_idx)
        if stacking
        else fn_variables_to_idx(model.fluent_to_idx),
        factor_to_idx(model.type_to_idx),
    )

    idx_func = fn_flatten_then_map_graph_to_idx(f) if stacking else f
    create_obs_dict_fn = fn_heterograph_to_heteroobs(idx_func)

    def graph_to_obsdata(g: HeteroGraph) -> HeteroIndexedFactorGraph:
        return create_obs_dict_fn(g)

    return graph_to_obsdata
