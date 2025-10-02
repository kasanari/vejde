import logging
import random
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np
from gymnasium.spaces import Dict
from numpy.typing import NDArray
import networkx as nx

from regawa.data.actions import ActionMask
from regawa.data.graph import Edges, GraphTypes, VariableDomain
from regawa.data.obs import Factors
from regawa.model import Grounding
from .grounding_utils import (
    arity,
    create_edges,
    predicate,
)
from regawa.data import (
    Edge,
    ObsData,
    Object,
    Variables,
    StringVariables,
)

logger = logging.getLogger(__name__)


def fn_graph_to_obsdata(
    rel_to_idx: Callable[[str], int],
    type_to_idx: Callable[[str], int],
):
    def map_graph_to_idx(
        g: GraphTypes,
        var_val_dtype: type,
    ) -> ObsData[VariableDomain]:
        arr = np.asarray
        factor_type_idx = arr(
            [type_to_idx(f_type) for f_type in g.factor_types], dtype=np.int64
        )
        idx_global_vars = arr(
            [rel_to_idx(p) for p in g.global_variables.types], dtype=np.int64
        )
        idx_vars = arr([rel_to_idx(p) for p in g.variables.types], dtype=np.int64)

        return ObsData(
            var=Variables(
                idx_vars,
                arr(g.variables.values, dtype=var_val_dtype),
                arr(g.variables.length),
                n_variable=g.variables.n_variable,
            ),
            factor=Factors(
                factor_type_idx,
                factor_type_idx.shape[0],  # number of factors
            ),
            edges=Edges(
                g.senders,
                g.receivers,
                arr(g.edge_attributes, dtype=np.int64),
            ),
            global_var=Variables(
                idx_global_vars,
                arr(g.global_variables.values, dtype=var_val_dtype),
                arr(g.global_variables.length),
                n_variable=g.global_variables.n_variable,
            ),
            action_masks=ActionMask(
                arr(g.action_type_mask, dtype=np.bool_),
                arr(g.action_arity_mask, dtype=np.bool_),
            ),
        )

    return map_graph_to_idx


def from_dict_action(
    action: tuple[str, ...],
    action_to_idx: Callable[[str], int],
    obj_to_idx: Callable[[str], int],
) -> tuple[int, ...]:
    action_idx = action_to_idx(action[0])
    object_idxs = [obj_to_idx(obj) for obj in action[1:]]
    return (action_idx, *object_idxs)


def idx_action_to_ground_value(
    action: Sequence[int],
    idx_to_action: Callable[[int], str],
    idx_to_obj: Callable[[int], str],
) -> Grounding:
    action_name = idx_to_action(action[0])
    o = tuple(idx_to_obj(obj_idx) for obj_idx in action[1:] if obj_idx != 0)
    return (action_name, *o)


def sample_action(action_space: Dict) -> dict[str, int]:
    action = action_space.sample()  # type: ignore
    chosen_action, value = random.choice(list(action.items()))  # type: ignore
    return {chosen_action: value}


def translate_edges(
    source_to_index: Callable[[Grounding], int],
    target_to_index: Callable[[str], int],
    edges: list[Edge],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    senders = np.asarray([source_to_index(edge[0]) for edge in edges], dtype=np.int64)
    receivers = np.asarray([target_to_index(edge[1]) for edge in edges], dtype=np.int64)
    return senders, receivers


def edge_attr(edges: Iterable[Edge]) -> Sequence[int]:
    return [edge[2] for edge in edges]


def object_list(
    obs_keys: Sequence[Grounding],
    objects_with_type: Callable[[Grounding], Sequence[Object]],
) -> Sequence[Object]:
    unique_objects = {obj for key in obs_keys for obj in objects_with_type(key)}
    # sorted_objects = unique_objects
    return [Object("None", "None")] + list(unique_objects)


def generate_bipartite_obs_func(
    cls: type[GraphTypes],
    action_fluent_type_mask: Callable[[str], tuple[bool, ...]],
    action_fluent_arity_mask: Callable[[str], tuple[bool, ...]],
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
        factor_node_values = [observations[g] for g in non_nullary_groundings]

        lengths = [len(x) if isinstance(x, Sequence) else 1 for x in factor_node_values]

        factor_node_predicates = [predicate(g) for g in non_nullary_groundings]

        object_names = [obj.name for obj in object_nodes]
        object_types = [obj.type for obj in object_nodes]
        object_indices = {name: idx for idx, name in enumerate(object_names)}

        edges = create_edges(non_nullary_groundings.keys())
        senders, receivers = translate_edges(
            lambda x: non_nullary_groundings[x], lambda x: object_indices[x], edges
        )

        # calculate grounding-factor distance matrix
        # use networkx
        G = nx.Graph()
        for edge in edges:
            G.add_edge(str(edge[0]), str(edge[1]))
        distances = dict(nx.shortest_path_length(G))
        distance_list = [
            (
                non_nullary_groundings[i],
                object_indices[j.name],
                distances[str(i)][j.name],
            )
            for i in non_nullary_groundings
            for j in object_nodes
            if j.name in distances[str(i)]
        ]

        pass

        action_type_mask = [
            action_fluent_type_mask(obj_type) for obj_type in object_types
        ]
        action_arity_mask = [
            action_fluent_arity_mask(obj_type) for obj_type in object_types
        ]

        global_vals = [observations[g] for g in nullary_groundings]
        global_lengths = [len(x) if isinstance(x, Sequence) else 1 for x in global_vals]

        if edges:
            assert senders.max() < len(
                factor_node_values
            ), "Senders index out of bounds."
            assert receivers.max() < len(object_types), "Receivers index out of bounds."

        return cls(
            StringVariables[VariableDomain](  # type: ignore
                factor_node_predicates,
                factor_node_values,
                lengths,
                len(non_nullary_groundings),
            ),
            object_names,
            object_types,
            senders,
            receivers,
            edge_attr(edges),
            StringVariables[VariableDomain](  # type: ignore
                [predicate(g) for g in nullary_groundings],
                global_vals,
                global_lengths,
                len(nullary_groundings),
            ),
            action_type_mask,
            action_arity_mask,
            list(non_nullary_groundings.keys()),
            nullary_groundings,
        )

    return f
