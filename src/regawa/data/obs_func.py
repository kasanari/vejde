from collections.abc import Callable
from regawa.data.graph import (
    Factors,
    StringFactorGraph,
    StringFactors,
    StringVariables,
    VariableDomain,
    Variables,
)
from regawa.data.graph_func import GraphTypes
from regawa.data.heterograph import HeteroGraph
from .obs import HeteroObsData, ObsData
import numpy as np


def fn_graph_to_obsdata(
    variables_to_idx: Callable[
        [StringVariables[VariableDomain], type], Variables[VariableDomain]
    ],
    factor_to_idx: Callable[[StringFactors], Factors],
):
    def map_graph_to_idx(
        g: StringFactorGraph[VariableDomain],
        var_val_dtype: type,
    ) -> ObsData[VariableDomain]:
        return ObsData(
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
            type,
        ],
        ObsData[VariableDomain],
    ],
):
    """
    Returns a function that takes a HeteroGraph and returns a HeteroObsData (for use in GNNs).
    """

    def heterograph_to_heteroobs(heterogenous_graph: HeteroGraph) -> HeteroObsData:
        return HeteroObsData(
            bool=fn_graph_to_idx(
                heterogenous_graph.boolean,  # type: ignore
                np.int8,
            ),
            float=fn_graph_to_idx(
                heterogenous_graph.numeric,  # type: ignore
                np.float32,
            ),
        )

    return heterograph_to_heteroobs
