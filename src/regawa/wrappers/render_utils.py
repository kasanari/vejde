from collections.abc import Callable

import numpy as np

from regawa.model import GroundValue
from regawa.wrappers.util_types import FactorGraph, RenderGraph


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


def create_render_graph(
    bool_g: FactorGraph[bool], numeric_g: FactorGraph[float]
) -> RenderGraph:
    def format_label(key: GroundValue) -> str:
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
