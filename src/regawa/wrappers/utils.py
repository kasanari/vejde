from collections import deque
from itertools import chain
import random
from typing import Any, NamedTuple, TypeVar
from collections.abc import Callable
import numpy as np
import logging
from gymnasium.spaces import Dict

from regawa import BaseModel

logger = logging.getLogger(__name__)

GroundValue = tuple[str, ...]


class Object(NamedTuple):
    name: str
    type: str


class Edge(NamedTuple):
    grounding: GroundValue
    object: str
    pos: int


class RenderGraph(NamedTuple):
    variable_labels: list[str]
    factor_labels: list[str]
    senders: np.ndarray[np.int64, Any]
    receivers: np.ndarray[np.int64, Any]
    edge_attributes: list[int]
    global_variables: list[str]


V = TypeVar("V")


class Variables[V](NamedTuple):
    types: np.ndarray[np.int64, Any]
    values: np.ndarray[V, Any]
    lengths: np.ndarray[np.int64, Any]


class IdxFactorGraph[V](NamedTuple):
    variables: Variables[V]
    factors: np.ndarray[np.int64, Any]
    senders: np.ndarray[np.int64, Any]
    receivers: np.ndarray[np.int64, Any]
    edge_attributes: np.ndarray[np.int64, Any]
    global_vars: Variables[V]
    action_mask: np.ndarray[np.bool_, Any]


class FactorGraph[V](NamedTuple):
    variables: list[str]
    variable_values: list[V]
    factors: list[str]
    factor_values: list[str]
    senders: np.ndarray[np.int64, Any]
    receivers: np.ndarray[np.int64, Any]
    edge_attributes: list[int]
    global_variables: list[str]
    global_variable_values: list[V]
    action_mask: np.ndarray[np.bool_, Any]


class StackedFactorGraph[V](NamedTuple):
    variables: list[str]
    variable_values: list[list[V]]
    factors: list[str]
    factor_values: list[str]
    senders: np.ndarray[np.int64, Any]
    receivers: np.ndarray[np.int64, Any]
    edge_attributes: list[int]
    global_variables: list[str]
    global_variable_values: list[list[V]]
    action_mask: np.ndarray[np.bool_, Any]


class HeteroGraph(NamedTuple):
    numeric: FactorGraph[float] | StackedFactorGraph[float]
    boolean: FactorGraph[bool] | StackedFactorGraph[bool]


def graph_to_dict[V](idx_g: IdxFactorGraph[V]) -> dict[str, Any]:
    return {
        "var_type": idx_g.variables.types,
        "var_value": idx_g.variables.values,
        "factor": idx_g.factors,
        "senders": idx_g.senders,
        "receivers": idx_g.receivers,
        "edge_attr": idx_g.edge_attributes,
        "length": idx_g.variables.lengths,
        "global_vars": idx_g.global_vars.types,
        "global_vals": idx_g.global_vars.values,
        "global_length": idx_g.global_vars.lengths,
        "action_mask": idx_g.action_mask,
        # "numeric": numeric,
    }


def create_render_graph(
    bool_g: FactorGraph[bool], numeric_g: FactorGraph[float]
) -> RenderGraph:
    boolean_labels = [
        f"{key}={bool_g.variable_values[idx]}"
        for idx, key in enumerate(bool_g.variables)
    ]
    numeric_labels = [
        f"{key}={numeric_g.variable_values[idx]}"
        for idx, key in enumerate(numeric_g.variables)
    ]

    labels = boolean_labels + numeric_labels

    factor_labels = [f"{key}" for key in bool_g.factors]

    edge_attributes = bool_g.edge_attributes + numeric_g.edge_attributes

    senders = np.concatenate(
        [bool_g.senders, numeric_g.senders + len(bool_g.variables)]
    )

    receivers = np.concatenate([bool_g.receivers, numeric_g.receivers])

    global_numeric = [
        f"{key}={numeric_g.global_variable_values[idx]}"
        for idx, key in enumerate(numeric_g.global_variables)
    ]
    global_boolean = [
        f"{key}={bool_g.global_variable_values[idx]}"
        for idx, key in enumerate(bool_g.global_variables)
    ]
    global_labels = global_boolean + global_numeric

    return RenderGraph(
        labels, factor_labels, senders, receivers, edge_attributes, global_labels
    )


def numeric_groundings(
    groundings: list[GroundValue], fluent_range: Callable[[str], type]
) -> list[GroundValue]:
    return [
        g
        for g in groundings
        if (fluent_range(predicate(g)) is float or fluent_range(predicate(g)) is int)
    ]


def bool_groundings(
    groundings: list[GroundValue], fluent_range: Callable[[str], type]
) -> list[GroundValue]:
    return [g for g in groundings if fluent_range(predicate(g)) is bool]


def create_graphs(
    rddl_obs: dict[GroundValue, Any],
    model: BaseModel,
):
    filtered_groundings = sorted(
        [
            g
            for g in rddl_obs
            if rddl_obs[g] is not None  # type: ignore
        ]
    )

    filtered_obs: dict[GroundValue, Any] = {k: rddl_obs[k] for k in filtered_groundings}

    bool_ground = bool_groundings(filtered_groundings, model.fluent_range)

    bool_g = generate_bipartite_obs(
        FactorGraph[bool],
        filtered_obs,
        bool_ground,
        model.fluent_param,
        model.num_actions,
    )

    n_g = numeric_groundings(filtered_groundings, model.fluent_range)

    numeric_g = generate_bipartite_obs(  # TODO add this to obs
        FactorGraph[float],
        filtered_obs,
        n_g,
        model.fluent_param,
        model.num_actions,
    )

    assert isinstance(bool_g, FactorGraph)
    assert isinstance(numeric_g, FactorGraph)
    # hetero_g = HeteroGraph(numeric_g, bool_g)
    return HeteroGraph(numeric_g, bool_g), filtered_groundings


def create_obs_dict(
    heterogenous_graph: HeteroGraph,
    model: BaseModel,
) -> dict[str, Any]:
    obs_boolean = graph_to_dict(
        map_graph_to_idx(
            heterogenous_graph.boolean,  # type: ignore
            model.fluent_to_idx,
            model.type_to_idx,
            np.int8,
        ),
    )

    obs_numeric = graph_to_dict(
        map_graph_to_idx(
            heterogenous_graph.numeric,  # type: ignore
            model.fluent_to_idx,
            model.type_to_idx,
            np.float32,
        )
    )

    obs = {
        "bool": obs_boolean,
        "float": obs_numeric,
    }

    return obs


def create_stacked_graphs(
    rddl_obs: dict[GroundValue, Any],
    model: BaseModel,
):
    filtered_groundings = sorted([g for g in rddl_obs])

    filtered_obs = {k: rddl_obs[k] for k in filtered_groundings}

    bool_ground = bool_groundings(filtered_groundings, model.fluent_range)

    bool_g = generate_bipartite_obs(
        StackedFactorGraph[bool],
        filtered_obs,
        bool_ground,
        model.fluent_param,
        model.num_actions,
    )

    n_g = numeric_groundings(filtered_groundings, model.fluent_range)

    numeric_g = generate_bipartite_obs(
        StackedFactorGraph[float],
        filtered_obs,
        n_g,
        model.fluent_param,
        model.num_actions,
    )

    assert isinstance(
        bool_g, StackedFactorGraph
    ), f"expected StackedFactorGraph but got {type(bool_g)}"
    assert isinstance(
        numeric_g, StackedFactorGraph
    ), f"expected StackedFactorGraph but got {type(numeric_g)}"
    return HeteroGraph(numeric_g, bool_g), filtered_groundings


def create_stacked_obs[V](
    heterograph: HeteroGraph,
    model: BaseModel,
) -> dict[str, Any]:
    obs_boolean = graph_to_dict(
        map_stacked_graph_to_idx(  # type: ignore
            heterograph.boolean,  # type: ignore
            model.fluent_to_idx,
            model.type_to_idx,
            np.int64,
        )
    )

    obs_numeric = graph_to_dict(
        map_stacked_graph_to_idx(  # type: ignore
            heterograph.numeric,  # type: ignore
            model.fluent_to_idx,
            model.type_to_idx,
            np.float32,
        )
    )

    obs = {
        "bool": obs_boolean,
        "float": obs_numeric,
    }

    return obs


def num_edges(arities: Callable[[str], int], groundings: list[GroundValue]) -> int:
    return sum(arities(predicate(g)) for g in groundings)


def from_dict_action(
    action: tuple[str, ...],
    action_to_idx: Callable[[str], int],
    obj_to_idx: Callable[[str], int],
) -> tuple[int, ...]:
    return tuple([action_to_idx(action[0])] + [obj_to_idx(obj) for obj in action[1:]])


def has_valid_parameters(
    action: GroundValue,
    obj_to_type: Callable[[str], str],
    fluent_params: Callable[[str], tuple[str, ...]],
) -> tuple[bool, str]:
    action_fluent = predicate(action)
    param_types = fluent_params(action_fluent)
    params: tuple[str, ...] = objects(action)

    if len(param_types) != len(params):
        return False, f"Expected {len(param_types)} parameters, got {len(params)}"

    for i, intended_param in enumerate(param_types):
        if intended_param != obj_to_type(params[i]):
            return False, f"Expected {intended_param}, got {obj_to_type(params[i])}"

    return True, ""


def to_dict_action(
    action: GroundValue,
    obj_to_type: Callable[[str], str],
    fluent_params: Callable[[str], tuple[str, ...]],
) -> dict[GroundValue, int]:
    action_fluent = predicate(action)
    has_valid_param, reason = has_valid_parameters(action, obj_to_type, fluent_params)
    action_fluent = "None" if not has_valid_param else action_fluent

    if not has_valid_param:
        logger.warning(f"Invalid action {action} because {reason}")

    action_dict = {} if action_fluent == "None" else {action: 1}

    return action_dict


def idx_action_to_ground_value(
    action: tuple[int, ...],
    idx_to_action: Callable[[int], str],
    idx_to_obj: Callable[[int], str],
) -> GroundValue:
    return (idx_to_action(action[0]),) + tuple(
        [idx_to_obj(obj) for obj in action[1:] if obj != 0]
    )


def sample_action(action_space: Dict) -> dict[str, int]:
    action = action_space.sample()  # type: ignore

    action: tuple[str, int] = random.choice(list(action.items()))  # type: ignore
    a = {action[0]: action[1]}
    return a


def predicate(key: GroundValue) -> str:
    return key[0]


def objects(key: GroundValue) -> tuple[str, ...]:
    return key[1:]


def objects_with_type(
    key: GroundValue, relation_to_types: Callable[[str, int], str]
) -> list[Object]:
    p = predicate(key)
    os = objects(key)
    return [Object(o, relation_to_types(p, i)) for i, o in enumerate(os)]


def arity(grounding: GroundValue) -> int:
    o = objects(grounding)
    return len(o)


def translate_edges(
    source_symbols: list[GroundValue], target_symbols: list[str], edges: list[Edge]
):
    return (
        np.array(
            [source_symbols.index(key[0]) for key in edges],
            dtype=np.int64,
        ),
        np.array([target_symbols.index(key[1]) for key in edges], dtype=np.int64),
    )


def edge_attr(edges: set[Edge]) -> np.ndarray[np.uint, Any]:
    return np.array([key[2] for key in edges], dtype=np.int64)


def create_edges(d: list[GroundValue]) -> list[Edge]:
    edges: deque[Edge] = deque()
    keys = sorted(d)
    for key in keys:
        for pos, object in enumerate(objects(key)):
            new_key = Edge(key, object, pos)
            edges.append(new_key)
    return list(edges)


def map_graph_to_idx[V](
    factorgraph: FactorGraph[V],
    rel_to_idx: Callable[[str], int],
    type_to_idx: Callable[[str], int],
    var_val_dtype: type,
) -> IdxFactorGraph[V]:
    edge_attributes = np.asarray(factorgraph.edge_attributes, dtype=np.int64)

    vals = np.array(  # type: ignore
        factorgraph.variable_values, dtype=var_val_dtype
    )  # TODO: handle None values in a better way

    global_vars_values = np.array(  # type: ignore
        factorgraph.global_variable_values, dtype=var_val_dtype
    )
    global_vars = np.array(
        [rel_to_idx(key) for key in factorgraph.global_variables], dtype=np.int64
    )

    return IdxFactorGraph(
        Variables(
            np.array(
                [rel_to_idx(key) for key in factorgraph.variables], dtype=np.int64
            ),
            vals,
            lengths=np.ones_like(vals, dtype=np.int64),
        ),
        np.array(
            [type_to_idx(object) for object in factorgraph.factor_values],
            dtype=np.int64,
        ),
        factorgraph.senders,
        factorgraph.receivers,
        edge_attributes,
        Variables(
            global_vars,
            global_vars_values,
            lengths=np.ones_like(global_vars_values, dtype=np.int64),
        ),
        factorgraph.action_mask,
    )


def map_stacked_graph_to_idx[V](
    factorgraph: StackedFactorGraph[V],
    rel_to_idx: Callable[[str], int],
    type_to_idx: Callable[[str], int],
    val_dtype: type,
) -> IdxFactorGraph[V]:
    edge_attributes = np.asarray(factorgraph.edge_attributes, dtype=np.int64)

    # Flatten the list of node history lists to account for different node history lengths
    vals = list(chain(*factorgraph.variable_values))

    vars = [
        [factorgraph.variables[i] for _ in v]
        for i, v in enumerate(factorgraph.variable_values)
    ]
    vars = list(chain(*vars))

    vars = [rel_to_idx(p) for p in vars]

    factors = [type_to_idx(object) for object in factorgraph.factor_values]

    global_vars_values = list(chain(*factorgraph.global_variable_values))

    global_vars = [
        [factorgraph.global_variables[i] for _ in v]
        for i, v in enumerate(factorgraph.global_variable_values)
    ]

    global_vars = list(chain(*global_vars))
    global_vars = [rel_to_idx(p) for p in global_vars]

    global_lengths = np.array(
        [len(v) for v in factorgraph.global_variables], dtype=np.int64
    )

    lengths = np.array([len(v) for v in factorgraph.variable_values], dtype=np.int64)

    return IdxFactorGraph(
        Variables(
            np.array(vars, dtype=np.int64),
            np.array(vals, dtype=val_dtype),
            lengths,
        ),
        np.array(
            factors,
            dtype=np.int64,
        ),
        factorgraph.senders,
        factorgraph.receivers,
        edge_attributes,
        Variables(
            np.array(
                global_vars,
            ),
            np.array(global_vars_values, dtype=val_dtype),
            global_lengths,
        ),
        action_mask=factorgraph.action_mask,
    )


def object_list(
    keys: list[GroundValue], relation_to_types: Callable[[str, int], str]
) -> list[Object]:
    obs_objects: set[Object] = set()
    for key in keys:
        for object in objects_with_type(key, relation_to_types):
            obs_objects.add(object)
    object_list = sorted(obs_objects)
    object_list = [Object("None", "None")] + object_list
    return object_list


def object_list_from_groundings(groundings: list[GroundValue]) -> list[str]:
    return sorted(set(chain(*[objects(g) for g in groundings])))


def predicate_list_from_groundings(groundings: list[GroundValue]) -> list[str]:
    return sorted(set(chain(*[predicate(g) for g in groundings])))


T = TypeVar(
    "T",
    type[FactorGraph[bool]],
    type[StackedFactorGraph[bool]],
    type[FactorGraph[float]],
    type[StackedFactorGraph[float]],
)


def generate_bipartite_obs(
    cls: T,
    obs: dict[GroundValue, bool | float],
    groundings: list[GroundValue],
    relation_to_types: Callable[[str, int], str],
    num_actions: int,
    # variable_ranges: dict[str, str],
) -> T:
    nullary_groundings = [g for g in groundings if arity(g) == 0]
    non_nullary_groundings = [g for g in groundings if arity(g) > 0]

    edges = create_edges(non_nullary_groundings)

    obj_list = object_list(
        obs.keys(), relation_to_types
    )  # NOTE: this makes factors common between boolean and numeric

    object_types = [object.type for object in obj_list]

    fact_node_values = [obs[key] for key in non_nullary_groundings]

    fact_node_predicate = [predicate(key) for key in non_nullary_groundings]

    obj_names = [object.name for object in obj_list]

    senders, receivers = translate_edges(
        non_nullary_groundings, obj_names, edges
    )  # edges are (var, factor)

    edge_attributes = [key[2] for key in edges]

    global_variables = [predicate(key) for key in nullary_groundings]
    global_variable_values = [obs[key] for key in nullary_groundings]

    action_mask = np.ones(
        (len(obj_names), num_actions), dtype=np.bool_
    )  # TODO make these actually mask the nodes

    if edges:
        assert max(senders) < len(fact_node_values)
        assert max(senders) < len(fact_node_predicate)
        assert max(receivers) < len(object_types)

    return cls(
        fact_node_predicate,
        fact_node_values,  # type: ignore
        obj_names,
        object_types,
        senders,
        receivers,
        edge_attributes,
        global_variables,
        global_variable_values,  # type: ignore
        action_mask,
    )


def to_graphviz(
    fg: RenderGraph,
    scaling: int = -20,
    # numeric,
):
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta"]
    graph = "graph G {\n"
    graph += f"overlap_scaling={scaling}\n"
    first_mapping = {}
    second_mapping = {}
    global_idx = 0
    for idx, label in enumerate(fg.variable_labels):
        graph += f'"{global_idx}" [label="{label}"]\n'
        first_mapping[idx] = global_idx
        global_idx += 1
    for idx, label in enumerate(fg.factor_labels):
        graph += f'"{global_idx}" [label="{label}", shape=box]\n'
        second_mapping[idx] = global_idx
        global_idx += 1
    for idx, label in enumerate(fg.global_variables):
        graph += f'"{global_idx}" [label="{label}", shape=diamond]\n'
        global_idx += 1

    for attribute, sender, receiver in zip(
        fg.edge_attributes, fg.senders, fg.receivers
    ):
        graph += f'"{first_mapping[sender]}" -- "{second_mapping[receiver]}" [color="{colors[int(attribute)]}"]\n'
    graph += "}"
    return graph


def to_graphviz_alt(
    predicate_node_classes: list[int],
    predicate_node_values: list[int],
    object_nodes: list[int],
    edges: list[tuple[int, int]],
    edge_attributes: list[int],
    idx_to_type: Callable[[int], str],
    idx_to_rel: Callable[[int], str],
) -> str:
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta"]
    graph = "graph G {\n"
    graph += "overlap_scaling=-20\n"
    first_mapping = {}
    second_mapping = {}
    global_idx = 0
    for idx, n_class in enumerate(predicate_node_classes):
        label = f'"{idx_to_rel(int(n_class))}={bool(predicate_node_values[idx])}"'
        graph += f'"{global_idx}" [label={label}]\n'
        first_mapping[idx] = global_idx
        global_idx += 1
    for idx, data in enumerate(object_nodes):
        graph += f'"{global_idx}" [label="{idx_to_type(data)}", shape=box]\n'
        second_mapping[idx] = global_idx
        global_idx += 1
    for attribute, edge in zip(edge_attributes, edges):
        graph += f'"{first_mapping[edge[0]]}" -- "{second_mapping[edge[1]]}" [color="{colors[attribute]}"]\n'
    graph += "}"
    return graph
