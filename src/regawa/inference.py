from collections.abc import Callable
from regawa.gnn.agent_utils import ActionMode
from regawa.gnn.gnn_agent import GraphAgent
from regawa.model.base_model import BaseModel
from typing import Any, NamedTuple
from torch import Tensor
from regawa import GroundValue
from regawa.wrappers.graph_utils import create_graphs_func, create_obs_dict_func
from regawa.wrappers.render_utils import create_render_graph
from regawa.wrappers.util_types import HeteroGraph, RenderGraph
import torch

GroundObs = dict[GroundValue, Any]


class AgentOutput(NamedTuple):
    action: tuple[str, str]
    weight_by_object: dict[str, float]
    weight_by_action: dict[str, float]
    graph: RenderGraph


def tensor_to_list(x: Tensor) -> list[float]:
    return list(x.squeeze().detach().cpu().numpy())  # type: ignore


def groundobs_to_graph(
    model: BaseModel,
    wrapper_func: Callable[[GroundObs], GroundObs],
):
    create_graph_fn = create_graphs_func(model)

    def f(obs: GroundObs):
        return create_graph_fn(wrapper_func(obs))[0]

    return f


def graph_to_obsdata(model: BaseModel):
    create_obs_dict_fn = create_obs_dict_func(model)

    def f(g: HeteroGraph):
        return create_obs_dict_fn(g)

    return f


# TODO make this work for node to action
@torch.inference_mode()
def get_agent_output_fn(
    agent: GraphAgent,
    model: BaseModel,
    wrapper_func: Callable[[GroundObs], GroundObs],
    action_mode: ActionMode,
    deterministic: bool = True,
):
    obs_to_graph = groundobs_to_graph(model, wrapper_func)
    graph_to_input = graph_to_obsdata(model)

    def action_then_node(
        o: GroundObs,
    ):
        g = obs_to_graph(o)
        r_g = create_render_graph(g.boolean, g.numeric)
        objs = r_g.factor_labels

        action, _, _, _, p_a, p_n__a = agent.sample_from_obs(
            graph_to_input(g), deterministic=deterministic
        )
        action = action.squeeze().detach().cpu().numpy()  # type: ignore

        weight_by_factor = {
            a: {
                k: float(v)
                for k, v in zip(objs, tensor_to_list(p_n__a[:, i]))
                if v > 0.0
            }
            for i, a in enumerate(model.action_fluents)
        }

        weight_by_action = {
            k: float(v)
            for k, v in zip(
                model.action_fluents,
                tensor_to_list(p_a),
            )
            if v > 0.0
        }

        return AgentOutput(
            action=(model.action_fluents[action[0]], objs[action[1]]),
            weight_by_object=weight_by_factor,
            weight_by_action=weight_by_action,
            graph=r_g,
        )

    def node_then_action(
        o: GroundObs,
    ):
        g = obs_to_graph(o)
        r_g = create_render_graph(g.boolean, g.numeric)
        objs = r_g.factor_labels

        action, _, _, _, p_n, p_a__n = agent.sample_from_obs(
            graph_to_input(g), deterministic=deterministic
        )
        action = action.squeeze().detach().cpu().numpy()  # type: ignore

        weight_by_factor = {
            k: float(v) for k, v in zip(objs, tensor_to_list(p_n)) if v > 0.0
        }

        weight_by_action = {
            o: {
                k: float(v)
                for k, v in zip(
                    model.action_fluents,
                    tensor_to_list(p_a__n[i, :]),
                )
                if v > 0.0
            }
            for i, o in enumerate(objs)
        }

        return AgentOutput(
            action=(model.action_fluents[action[0]], objs[action[1]]),
            weight_by_object=weight_by_factor,
            weight_by_action=weight_by_action,
            graph=r_g,
        )

    modes = {
        ActionMode.NODE_THEN_ACTION: node_then_action,
        ActionMode.ACTION_THEN_NODE: action_then_node,
    }

    return modes[action_mode]
