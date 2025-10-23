from collections.abc import Callable, Sequence
from itertools import chain


from regawa.data import ObsData, StackedStringFactorGraph
from regawa.data.graph import GraphTypes, StringVariables, VariableDomain


def flatten(
    vals: Sequence[Sequence[VariableDomain]], vars: Sequence[str]
) -> StringVariables[VariableDomain]:
    # Flatten the list of node history lists to account for different node history lengths
    flat_vals = list(chain(*vals))
    v = [[vars[i] for _ in v] for i, v in enumerate(vals)]  # expand the variable names
    flat_vars = list(chain(*v))
    lengths = [len(v) for v in vals]  # lengths of each variable history
    n_variable = len(lengths)  # number of unique variables
    return StringVariables(flat_vars, flat_vals, lengths, n_variable=n_variable)


def flatten_values(
    factorgraph: StackedStringFactorGraph[VariableDomain],
) -> tuple[StringVariables[VariableDomain], StringVariables[VariableDomain]]:
    return (
        flatten(factorgraph.variables.values, factorgraph.variables.types),
        flatten(
            factorgraph.global_variables.values, factorgraph.global_variables.types
        ),
    )


def fn_flatten_map_graph_to_idx(
    map_graph_to_idx: Callable[[GraphTypes, type], ObsData[VariableDomain]],
):
    def flatten_map_graph_to_idx(
        factorgraph: StackedStringFactorGraph[VariableDomain],
        var_val_dtype: type,
    ) -> ObsData[VariableDomain]:
        flattened_graph = factorgraph._replace(
            variables=flatten(
                factorgraph.variables.values, factorgraph.variables.types
            ),
            global_variables=flatten(
                factorgraph.global_variables.values,
                factorgraph.global_variables.types,
            ),
        )

        return map_graph_to_idx(
            flattened_graph,
            var_val_dtype,
        )

    return flatten_map_graph_to_idx
