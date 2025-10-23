from collections.abc import Callable
from regawa import GroundObs
from regawa.data.obs import HeteroObsData
from regawa.wrappers.index_obs_wrapper import fn_idx_obs
from regawa.policy import ActionMode
from regawa.policy import GraphAgent
from regawa.model import BaseModel
from typing import NamedTuple
from torch import Tensor
from regawa.wrappers import fn_groundobs_to_heterograph
from regawa.wrappers import create_render_graph
from regawa.wrappers.render_utils import RenderGraph
import torch


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


@torch.inference_mode()
def fn_get_agent_output(
    agent: GraphAgent,
    model: BaseModel,
    wrapper_func: Callable[[GroundObs], GroundObs],
    action_mode: ActionMode,
    deterministic: bool = True,
    stacking: bool = False,
):
    obs_to_graph = fn_groundobs_to_heterograph(model, stacking)
    graph_to_input = fn_idx_obs(model, stacking=stacking)

    modes = {
        ActionMode.NODE_THEN_ACTION: fn_node_then_action,
        ActionMode.ACTION_THEN_NODE: fn_action_then_node,
    }

    fn = modes[action_mode](agent, model, deterministic)

    def get_agent_output(ground_obs: GroundObs):
        hetero_graph = obs_to_graph(wrapper_func(ground_obs))
        heteroobs = graph_to_input(hetero_graph)
        r_g = create_render_graph(hetero_graph.boolean, hetero_graph.numeric)
        return fn(heteroobs, r_g)

    return get_agent_output


def fn_action_then_node(
    agent: GraphAgent,
    model: BaseModel,
    deterministic: bool = True,
):
    def action_then_node(
        o: HeteroObsData,
        g: RenderGraph,
    ):
        objs = g.factor_labels

        action, _, _, _, p_a, p_n__a = agent.sample_from_obs(
            o, deterministic=deterministic
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
            graph=g,
        )

    return action_then_node


def fn_node_then_action(
    agent: GraphAgent,
    model: BaseModel,
    deterministic: bool = True,
):
    def node_then_action(
        o: HeteroObsData,
        g: RenderGraph,
    ):
        objs = g.factor_labels

        action, _, _, _, p_n, p_a__n = agent.sample_from_obs(
            o, deterministic=deterministic
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
            graph=g,
        )

    return node_then_action
