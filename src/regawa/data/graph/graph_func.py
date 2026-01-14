import itertools
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import cache

import numpy as np
from numpy.typing import NDArray

from regawa.model import (
    BaseModel,
    Grounding,
    fn_valid_action_fluents_given_arity,
    fn_valid_action_fluents_given_type,
    objects,
    predicate,
)

from .graph import (
    ActionMask,
    Edge,
    NullObject,
    Object,
    StringVariables,
    VariableDomain,
)

type StrToInt = Callable[[str], int]


def fn_objects_with_type(relation_to_types: Callable[[str, int], str]):
    """
    Returns a function that takes a grounding and returns a list of Objects (object name, type).
    """

    @cache
    def objects_with_type(
        key: Grounding,
    ) -> list[Object]:
        p = predicate(key)
        os = objects(key)
        return [Object(o, relation_to_types(p, i)) for i, o in enumerate(os)]

    return objects_with_type


def object_list(
    obs_keys: Sequence[Grounding],
    objects_with_type: Callable[[Grounding], Sequence[Object]],
) -> Sequence[Object]:
    unique_objects = {obj for key in obs_keys for obj in objects_with_type(key)}
    # sorted_objects = unique_objects
    return [NullObject, *unique_objects]


@cache
def get_edges(key: Grounding) -> list[Edge]:
    """
    Returns a list of edges for a given grounding.
    Each edge connects the predicate to one of its objects.
    An edge is represented as a tuple (predicate, object, position).
    """
    return [Edge(key, o, pos) for pos, o in enumerate(objects(key))]


def create_edges(d: Iterable[Grounding]) -> list[Edge]:
    edges = [get_edges(key) for key in d]
    return list(itertools.chain(*edges))


def translate_edges(
    source_to_index: Callable[[Grounding], int],
    target_to_index: Callable[[str], int],
    edges: list[Edge],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    senders = np.asarray([source_to_index(edge[0]) for edge in edges], dtype=np.int64)
    receivers = np.asarray([target_to_index(edge[1]) for edge in edges], dtype=np.int64)
    return senders, receivers


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


def edge_attr(edges: Iterable[Edge]) -> NDArray[np.int64]:
    return np.asarray([edge[2] for edge in edges], dtype=np.int64)


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
