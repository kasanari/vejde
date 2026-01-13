from collections.abc import Callable, Mapping, Sequence

from regawa.data.graph import (
    ActionMask,
    Edges,
    Object,
    StringFactors,
    VariableDomain,
    create_edges,
    create_variables,
    edge_attr,
    translate_edges,
)
from regawa.data.obs import GraphTypes
from regawa.model import Grounding, arity


def generate_bipartite_obs_func(
    cls: type[GraphTypes],
    action_mask_func: Callable[[Sequence[str]], ActionMask],
):
    def f(
        observations: Mapping[Grounding, VariableDomain],
        groundings: Sequence[Grounding],
        object_nodes: Sequence[Object],
    ) -> GraphTypes:
        nullary_groundings = [g for g in groundings if arity(g) == 0]
        non_nullary_groundings = {
            g: idx for idx, g in enumerate(g for g in groundings if arity(g) > 0)
        }

        object_names = [obj.name for obj in object_nodes]
        object_types = [obj.type for obj in object_nodes]
        object_indices = {o.name: idx for idx, o in enumerate(object_nodes)}

        edges = create_edges(non_nullary_groundings.keys())
        v_to_f, f_to_v = translate_edges(
            lambda x: non_nullary_groundings[x], lambda x: object_indices[x], edges
        )

        g = cls(
            create_variables(observations, non_nullary_groundings.keys()),  # type: ignore
            StringFactors(
                object_names,
                object_types,
            ),
            Edges(v_to_f, f_to_v, edge_attr(edges)),
            create_variables(observations, nullary_groundings),  # type: ignore
            action_mask_func(object_types),
        )

        if edges:
            assert v_to_f.max() < len(
                g.variables.values
            ), "Senders index out of bounds."
            assert f_to_v.max() < len(object_types), "Receivers index out of bounds."

        return g

    return f