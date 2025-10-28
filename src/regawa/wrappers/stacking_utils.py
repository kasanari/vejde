from collections.abc import Sequence
from itertools import chain


from regawa.data import StackedStringFactorGraph
from regawa.data.graph import (
    StringFactorGraph,
    StringVariables,
    VariableDomain,
)


def flatten(
    vals: Sequence[Sequence[VariableDomain]],
    vars: Sequence[str],
    groundings: Sequence[Grounding],
) -> StringVariables[VariableDomain]:
    # Flatten the list of node history lists to account for different node history lengths
    flat_vals = list(chain(*vals))
    v = [[vars[i] for _ in v] for i, v in enumerate(vals)]  # expand the variable names
    flat_vars = list(chain(*v))
    lengths = [len(v) for v in vals]  # lengths of each variable history
    n_variable = len(lengths)  # number of unique variables
    return StringVariables(
        flat_vars, flat_vals, lengths, n_variable=n_variable, groundings=groundings
    )


def flatten_values(
    factorgraph: StackedStringFactorGraph[VariableDomain],
) -> tuple[StringVariables[VariableDomain], StringVariables[VariableDomain]]:
    return (
        flatten(factorgraph.variables.values, factorgraph.variables.types),
        flatten(
            factorgraph.global_variables.values, factorgraph.global_variables.types
        ),
    )


def flatten_stacked_graph(
    factorgraph: StackedStringFactorGraph[VariableDomain],
) -> StringFactorGraph[VariableDomain]:
    return StringFactorGraph(
        variables=flatten(
            factorgraph.variables.values,
            factorgraph.variables.types,
            factorgraph.variables.groundings,
        ),
        global_variables=flatten(
            factorgraph.global_variables.values,
            factorgraph.global_variables.types,
            factorgraph.global_variables.groundings,
        ),
        factors=factorgraph.factors,
        factor_types=factorgraph.factor_types,
        v_to_f=factorgraph.v_to_f,
        f_to_v=factorgraph.f_to_v,
        edge_attributes=factorgraph.edge_attributes,
        action_masks=factorgraph.action_masks,
    )
