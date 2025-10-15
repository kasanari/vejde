from collections.abc import Callable
from regawa import GroundObs
from regawa.wrappers.index_obs_wrapper import fn_idx_obs
from regawa.data.graph import StringFactorGraph
from regawa.policy import ActionMode
from regawa.policy import GraphAgent
from regawa.model import BaseModel
from typing import NamedTuple
from torch import Tensor
from regawa.wrappers import fn_groundobs_to_heterograph
from regawa.wrappers import create_render_graph
from regawa.wrappers.render_utils import RenderGraph
import torch
import numpy as np
from regawa.data.graph import StackedStringFactorGraph


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

    def action_then_node(
        o: GroundObs,
    ):
        o = wrapper_func(o)
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
