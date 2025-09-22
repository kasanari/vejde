from collections.abc import Callable
from regawa import GroundObs
from regawa.wrappers.graph_utils import fn_regular_map_graph_to_idx
from regawa.wrappers.stacking_utils import fn_flatten_map_graph_to_idx
from regawa.wrappers.types import FactorGraph
from regawa.policy import ActionMode
from regawa.data import HeteroObsData
from regawa.policy import GraphAgent
from regawa.model import BaseModel
from typing import NamedTuple
from torch import Tensor
from regawa.wrappers import fn_obsdict_to_graph, fn_heterograph_to_heteroobs
from regawa.wrappers import create_render_graph
from regawa.wrappers import HeteroGraph, RenderGraph
import torch
import numpy as np
from regawa.wrappers.types import StackedFactorGraph


class NodeThenActionAgentOutput(NamedTuple):
    action: tuple[str, str]
    weight_by_object: dict[str, dict[str, float]]
    weight_by_action: dict[str, float]
    joint_probs: dict[tuple[str, str], float]
    graph: RenderGraph


class ActionThenNodeAgentOutput(NamedTuple):
    action: tuple[str, str]
    weight_by_object: dict[str, float]
    weight_by_action: dict[str, dict[str, float]]
    joint_probs: dict[tuple[str, str], float]
    graph: RenderGraph


def tensor_to_list(x: Tensor) -> list[float]:
    return list(x.squeeze().detach().cpu().numpy())  # type: ignore


def fn_groundobs_to_graph[
    T: FactorGraph[np.bool_] | StackedFactorGraph[np.bool_],
    S: FactorGraph[np.float32] | StackedFactorGraph[np.float32],
](
    bool_graph_cls: type[T],
    numeric_graph_cls: type[S],
    model: BaseModel,
    wrapper_func: Callable[[GroundObs], GroundObs],
) -> Callable[[GroundObs], HeteroGraph]:
    create_graph_fn = fn_obsdict_to_graph(model, bool_graph_cls, numeric_graph_cls)

    def groundobs_to_graph(obs: GroundObs) -> HeteroGraph:
        return create_graph_fn(wrapper_func(obs))

    return groundobs_to_graph


def fn_graph_to_obsdata(model: BaseModel, stacking: bool = False):
    idx_func = (
        fn_flatten_map_graph_to_idx(
            model.fluent_to_idx,
            model.type_to_idx,
        )
        if stacking
        else fn_regular_map_graph_to_idx(
            model.fluent_to_idx,
            model.type_to_idx,
        )
    )
    create_obs_dict_fn = fn_heterograph_to_heteroobs(idx_func)

    def graph_to_obsdata(g: HeteroGraph) -> HeteroObsData:
        return create_obs_dict_fn(g)

    return graph_to_obsdata


@torch.inference_mode()
def fn_get_agent_output(
    agent: GraphAgent,
    model: BaseModel,
    wrapper_func: Callable[[GroundObs], GroundObs],
    action_mode: ActionMode,
    deterministic: bool = True,
    stacking: bool = False,
):
    bool_graph_cls, numeric_graph_cls = (
        (StackedFactorGraph[np.bool_], StackedFactorGraph[np.float32])
        if stacking
        else (FactorGraph[np.bool_], FactorGraph[np.float32])
    )
    obs_to_graph = fn_groundobs_to_graph(
        bool_graph_cls, numeric_graph_cls, model, wrapper_func
    )
    graph_to_input = fn_graph_to_obsdata(model)

    def action_then_node(
        o: GroundObs,
    ):
        g = obs_to_graph(o)
        r_g = create_render_graph(g.boolean, g.numeric)
        objs = r_g.factor_labels

        action, _, _, _, p_a, p_n__a = agent.sample_from_obs(
            graph_to_input(g), deterministic=deterministic
        )
        action_tup: tuple[int, int] = tuple(action.squeeze().detach().cpu().numpy())  # type: ignore

        weight_by_factor = {
            a: {
                k: float(v)
                for k, v in zip(objs, tensor_to_list(p_n__a[:, i]))
                if v > 1e-4
            }
            for i, a in enumerate(model.action_fluents)
        }

        weight_by_action = {
            k: float(v)
            for k, v in zip(
                model.action_fluents,
                tensor_to_list(p_a),
            )
            if v > 1e-4
        }

        joint_probs = {
            (a, o): pa * po
            for a, pa in weight_by_action.items()
            for o, po in weight_by_factor[a].items()
            if (pa * po) > 1e-4
        }

        return NodeThenActionAgentOutput(
            action=(model.action_fluents[action_tup[0]], objs[action_tup[1]]),
            weight_by_object=weight_by_factor,
            weight_by_action=weight_by_action,
            joint_probs=joint_probs,
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
        action_tup: tuple[int, int] = tuple(action.squeeze().detach().cpu().numpy())  # type: ignore

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
                if v > 1e-4
            }
            for i, o in enumerate(objs)
        }

        joint_probs = {
            (a, o): po * pa
            for o, po in weight_by_factor.items()
            for a, pa in weight_by_action[o].items()
            if (po * pa) > 1e-4
        }

        return ActionThenNodeAgentOutput(
            action=(model.action_fluents[action_tup[0]], objs[action_tup[1]]),
            weight_by_object=weight_by_factor,
            weight_by_action=weight_by_action,
            joint_probs=joint_probs,
            graph=r_g,
        )

    modes = {
        ActionMode.NODE_THEN_ACTION: node_then_action,
        ActionMode.ACTION_THEN_NODE: action_then_node,
    }

    return modes[action_mode]
