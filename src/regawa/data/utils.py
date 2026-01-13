import logging
import random
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np
from gymnasium.spaces import Dict
from numpy.typing import NDArray

from regawa.data.actions import ActionMask
from regawa.data.graph import (
    Edges,
    Factors,
    StringFactorGraph,
    StringFactors,
    VariableDomain,
    NullObject,
    Edge,
    Object,
    StringVariables,
    Variables,
)
from regawa.data.graph_func import GraphTypes, create_edges
from regawa.data.obs import ObsData
from regawa.model import Grounding, BaseModel
from regawa.model.utils import (
    fn_valid_action_fluents_given_arity,
    fn_valid_action_fluents_given_type,
)
from regawa.model.grounding_func import (
    arity,
    predicate,
)

logger = logging.getLogger(__name__)


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
            zip(*variables.values) if variables.values else ([], [])
        )

        return Variables(
            arr([rel_to_idx(p) for p in variables.types], dtype=np.int64),
            arr(variable_values, dtype=var_val_dtype),
            arr(variables.length),
            n_variable=variables.n_variable,
            times=arr(times, dtype=np.int64),
        )

    return map_variables_to_idx


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
    o = tuple(idx_to_obj(obj_idx) for obj_idx in action[1:])
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


def edge_attr(edges: Iterable[Edge]) -> NDArray[np.int64]:
    return np.asarray([edge[2] for edge in edges], dtype=np.int64)


def object_list(
    obs_keys: Sequence[Grounding],
    objects_with_type: Callable[[Grounding], Sequence[Object]],
) -> Sequence[Object]:
    unique_objects = {obj for key in obs_keys for obj in objects_with_type(key)}
    # sorted_objects = unique_objects
    return [NullObject] + list(unique_objects)


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


def fn_action_masks(
    model: BaseModel,
):
    action_fluent_type_mask = fn_valid_action_fluents_given_type(model)
    action_fluent_arity_mask = fn_valid_action_fluents_given_arity(model)

    def f(
        object_types: Sequence[str],
    ) -> ActionMask:
        return ActionMask(
            np.array(
                tuple(map(action_fluent_type_mask, object_types)),
                dtype=np.bool_,
            ),  # n_object x n_actions
            np.array(
                tuple(map(action_fluent_arity_mask, object_types)),
                dtype=np.bool_,
            ),  # n_object x n_actions
        )

    return f


def create_variables(
    observations: Mapping[Grounding, VariableDomain],
    groundings: Sequence[Grounding],
) -> StringVariables[VariableDomain]:
    factor_node_values = [observations[g] for g in groundings]
    lengths = [len(x) if isinstance(x, Sequence) else 1 for x in factor_node_values]
    factor_node_predicates = [predicate(g) for g in groundings]
    return StringVariables[VariableDomain](  # type: ignore
        factor_node_predicates,
        factor_node_values,
        lengths,
        len(groundings),
        groundings,
    )
