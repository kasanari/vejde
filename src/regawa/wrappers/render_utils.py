from collections.abc import Callable
from itertools import chain

import numpy as np

from regawa.model import Grounding
from regawa.model.base_model import BaseModel
from regawa.wrappers.grounding_utils import create_edges, objects
from regawa.wrappers.types import (
    FactorGraph,
    RenderGraph,
    StackedFactorGraph,
)
from regawa.wrappers.utils import translate_edges


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


def to_graphviz(
    fg: RenderGraph,
    scaling: int = -20,
    pprint: bool = False,
    # numeric,
):
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta"]
    graph = "graph G {"
    graph += "\n" if pprint else " "
    graph += "overlap=false"
    graph += "\n" if pprint else " "
    first_mapping = {}
    second_mapping = {}
    global_idx = 0
    for idx, label in enumerate(fg.variable_labels):
        graph += f'"{global_idx}" [label="{label}"]'
        graph += "\n" if pprint else " "
        first_mapping[idx] = global_idx
        global_idx += 1
    for idx, label in enumerate(fg.factor_labels):
        graph += f'"{global_idx}" [label="{label}", shape=box]'
        graph += "\n" if pprint else " "
        second_mapping[idx] = global_idx
        global_idx += 1
    for idx, label in enumerate(fg.global_variables):
        graph += f'"{global_idx}" [label="{label}", shape=diamond]'
        graph += "\n" if pprint else " "
        global_idx += 1

    for attribute, sender, receiver in zip(
        fg.edge_attributes, fg.senders, fg.receivers
    ):
        graph += f'"{first_mapping[sender]}" -- "{second_mapping[receiver]}" [color="{colors[int(attribute)]}"]'
        graph += "\n" if pprint else " "
    graph += "}"
    return graph


def create_render_graph(
    bool_g: FactorGraph[np.bool_] | StackedFactorGraph[np.bool_],
    numeric_g: FactorGraph[np.float32] | StackedFactorGraph[np.float32],
) -> RenderGraph:
    def format_label(key: Grounding) -> str:
        fluent, *args = key
        return f"{fluent}({', '.join(args)})" if args else fluent

    boolean_labels = [
        f"{format_label(key)}={bool_g.variable_values[idx]}"
        for idx, key in enumerate(bool_g.groundings)
    ]
    numeric_labels = [
        f"{format_label(key)}={numeric_g.variable_values[idx]}"
        for idx, key in enumerate(numeric_g.groundings)
    ]

    labels = boolean_labels + numeric_labels

    factor_labels = [f"{key}" for key in bool_g.factors]

    edge_attributes = np.concatenate(
        (bool_g.edge_attributes, numeric_g.edge_attributes)
    )

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


def render_lifted(model: BaseModel):
    params = {p: model.fluent_params(p) for p in model.fluents}

    atoms: list[Grounding] = [(p, *params[p]) for p in model.fluents]
    global_vars = [a for a in atoms if len(objects(a)) == 0]
    non_global_vars = [a for a in atoms if len(objects(a)) > 0]

    edges = create_edges(non_global_vars)

    o = sorted(set(chain(*[objects(a) for a in non_global_vars])))

    senders, receivers = translate_edges(non_global_vars.index, o.index, edges)

    edge_attributes = [key[2] for key in edges]

    graph = FactorGraph(
        variables=list(map(str, non_global_vars)),
        variable_values=[np.bool_(True) for _ in non_global_vars],
        factors=o,
        factor_types=o,
        senders=senders,
        receivers=receivers,
        edge_attributes=edge_attributes,
        global_variables=list(map(str, global_vars)),
        global_variable_values=[np.bool_(True) for _ in global_vars],
        groundings=non_global_vars,
        global_groundings=global_vars,
        action_arity_mask=[(True,) for _ in o],
        action_type_mask=[(False,) for _ in o],
    )

    n_graph = FactorGraph[np.float32](
        variables=[],
        variable_values=[],
        factors=[],
        factor_types=[],
        senders=np.array([], dtype=np.int64),
        receivers=np.array([], dtype=np.int64),
        edge_attributes=[],
        global_variables=[],
        global_variable_values=[],
        groundings=[],
        global_groundings=[],
        action_arity_mask=[(True,) for _ in o],
        action_type_mask=[(False,) for _ in o],
    )

    render_g = create_render_graph(graph, n_graph)

    return to_graphviz(render_g, scaling=0)

    pass
