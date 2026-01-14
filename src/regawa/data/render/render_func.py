from collections.abc import Callable, Sequence
from itertools import chain

import numpy as np

from regawa.data.factor_graph import StringFactorGraph
from regawa.data.graph import (
    ActionMask,
    Edges,
    StringFactors,
    StringVariables,
    create_edges,
    translate_edges,
)
from regawa.data.stacked import StackedStringFactorGraph
from regawa.model import BaseModel, Grounding, objects

from .render_graph import RenderGraph


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
    for attribute, edge in zip(edge_attributes, edges, strict=False):
        graph += f'"{first_mapping[edge[0]]}" -- "{second_mapping[edge[1]]}" [color="{colors[attribute]}"]\n'
    graph += "}"
    return graph


def to_graphviz(
    fg: RenderGraph,
    pprint: bool = False,
    # numeric,
):
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta"]
    graph = "graph G {"
    graph += "\n" if pprint else " "
    graph += "overlap=false"
    graph += "\n" if pprint else " "
    v_mapping = {}
    f_mapping = {}
    global_idx = 0
    for idx, label in enumerate(fg.variable_labels):
        graph += f'"{global_idx}" [label="{label}"]'
        graph += "\n" if pprint else " "
        v_mapping[idx] = global_idx
        global_idx += 1
    for idx, label in enumerate(fg.factor_labels):
        graph += f'"{global_idx}" [label="{label}", shape=box]'
        graph += "\n" if pprint else " "
        f_mapping[idx] = global_idx
        global_idx += 1
    for _, label in enumerate(fg.global_variables):
        graph += f'"{global_idx}" [label="{label}", shape=diamond]'
        graph += "\n" if pprint else " "
        global_idx += 1

    for attribute, v, f in zip(fg.edge_attributes, fg.v_to_f, fg.f_to_v, strict=False):
        graph += (
            f'"{v_mapping[v]}" -- "{f_mapping[f]}" [color="{colors[int(attribute)]}"]'
        )
        graph += "\n" if pprint else " "
    graph += "}"
    return graph


def create_render_graph(
    bool_g: StringFactorGraph[np.int8] | StackedStringFactorGraph[np.int8],
    numeric_g: StringFactorGraph[np.float32] | StackedStringFactorGraph[np.float32],
) -> RenderGraph:
    def format_label(key: Grounding) -> str:
        fluent, *args = key
        return f"{fluent}({', '.join(args)})" if args else fluent

    boolean_labels = [
        f"{format_label(key)}={bool_g.variables.values[idx]}"
        for idx, key in enumerate(bool_g.variables.groundings)
    ]
    numeric_labels = [
        f"{format_label(key)}={numeric_g.variables.values[idx]}"
        for idx, key in enumerate(numeric_g.variables.groundings)
    ]

    labels = boolean_labels + numeric_labels

    factor_labels = [f"{key}" for key in bool_g.factors.names]

    edge_attributes: Sequence[int] = np.concatenate(
        (bool_g.edges.edge_attr, numeric_g.edges.edge_attr)
    )  # type: ignore

    v_to_f = np.concatenate(
        [bool_g.edges.v_to_f, numeric_g.edges.v_to_f + len(bool_g.variables.values)]
    )

    f_to_v = np.concatenate([bool_g.edges.f_to_v, numeric_g.edges.f_to_v])

    global_numeric = [
        f"{key}={numeric_g.global_variables.values[idx]}"
        for idx, key in enumerate(numeric_g.global_variables.values)
    ]
    global_boolean = [
        f"{key}={bool_g.global_variables.values[idx]}"
        for idx, key in enumerate(bool_g.global_variables.values)
    ]
    global_labels = global_boolean + global_numeric

    return RenderGraph(
        labels, factor_labels, v_to_f, f_to_v, edge_attributes, global_labels
    )


def render_lifted(model: BaseModel):
    params = {p: model.fluent_params(p) for p in model.fluents}

    atoms: list[Grounding] = [(p, *params[p]) for p in model.fluents]
    global_vars = [a for a in atoms if len(objects(a)) == 0]
    non_global_vars = [a for a in atoms if len(objects(a)) > 0]

    edges = create_edges(non_global_vars)

    o = sorted(set(chain(*[objects(a) for a in non_global_vars])))

    v_to_f, f_to_v = translate_edges(non_global_vars.index, o.index, edges)

    edge_attributes = np.array([key[2] for key in edges])

    graph = StringFactorGraph(
        StringVariables[np.bool_](
            list(map(str, non_global_vars)),
            [np.bool_(True) for _ in non_global_vars],
            [1 for _ in non_global_vars],
            n_variable=len(non_global_vars),
            groundings=non_global_vars,
        ),
        factors=StringFactors(
            types=o,
            names=o,
        ),
        edges=Edges(
            v_to_f=v_to_f,
            f_to_v=f_to_v,
            edge_attr=edge_attributes,
        ),
        global_variables=StringVariables[np.bool_](
            list(map(str, global_vars)),
            [np.bool_(True) for _ in global_vars],
            [1 for _ in global_vars],
            n_variable=len(global_vars),
            groundings=global_vars,
        ),
        action_masks=ActionMask(
            action_arity_mask=np.array([(True,) for _ in o]),
            action_type_mask=np.array([(False,) for _ in o]),
        ),
    )

    n_graph = StringFactorGraph[np.float32](
        variables=StringVariables[np.float32]([], [], [], 0, []),
        factors=StringFactors(
            names=[],
            types=[],
        ),
        edges=Edges(
            v_to_f=np.array([], dtype=np.int64),
            f_to_v=np.array([], dtype=np.int64),
            edge_attr=np.array([]),
        ),
        global_variables=StringVariables[np.float32]([], [], [], 0, []),
        action_masks=ActionMask(
            action_arity_mask=np.array([(True,) for _ in o]),
            action_type_mask=np.array([(False,) for _ in o]),
        ),
    )

    render_g = create_render_graph(graph, n_graph)

    return to_graphviz(render_g, scaling=0)


