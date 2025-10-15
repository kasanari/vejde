import logging
from typing import Any, SupportsFloat
import gymnasium as gym
from regawa import BaseModel, GroundObs, Grounding
from ..data.graph import HeteroGraph, StackedStringFactorGraph
from regawa.model import StackedGroundObs
from .graph_utils import fn_groundobs_to_heterograph
from .render_utils import create_render_graph, to_graphviz
from .render_utils import RenderGraph
import numpy as np

logger = logging.getLogger(__name__)


class StackingGroundedGraphWrapper(
    gym.Wrapper[
        HeteroGraph,
        GroundObs | tuple[int, ...],
        StackedGroundObs,
        GroundObs | tuple[int, ...],
    ]
):
    def __init__(
        self,
        env: gym.Env[StackedGroundObs, GroundObs | tuple[int, ...]],
        model: BaseModel,
        render_mode: str = "human",
        add_render_graph_to_info: bool = True,
    ) -> None:
        super().__init__(env)
        self.model = model
        self.last_action: Grounding | None = None
        self.last_g: RenderGraph | None = None
        self._object_to_type: dict[str, str] = {"None": "None"}
        self.create_graphs = fn_groundobs_to_heterograph(
            model,
            stacking=True,
        )

        self.add_render_graph_to_info = add_render_graph_to_info

    def render(self):
        return to_graphviz(self.last_g, scaling=10) if self.last_g is not None else None

    def _prepare_info(
        self,
        rddl_obs: GroundObs,
        graph: HeteroGraph,
        add_render_graph_to_info: bool = False,
    ) -> tuple[dict[str, Any], RenderGraph | None]:
        combined_graph = (
            create_render_graph(graph.boolean, graph.numeric)
            if add_render_graph_to_info
            else None
        )

        info: dict[str, Any] = {
            "state": combined_graph,
            "rddl_obs": rddl_obs,
            "action_fluents": self.model.action_fluents,
        }
        return info, combined_graph

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[HeteroGraph, dict[str, Any]]:
        rddl_obs, info = self.env.reset(seed=seed)
        graph = self.create_graphs(rddl_obs)  # type: ignore
        info_update, combined_graph = self._prepare_info(
            rddl_obs,  # type: ignore
            graph,
            self.add_render_graph_to_info,
        )
        info = info | info_update
        info["rddl_state"] = (
            self.env.unwrapped.state if hasattr(self.env.unwrapped, "state") else {}  # type: ignore
        )

        self.last_g = combined_graph

        return graph, info

    def step(
        self, action: GroundObs | tuple[int, ...]
    ) -> tuple[HeteroGraph, SupportsFloat, bool, bool, dict[str, Any]]:
        rddl_obs, reward, terminated, truncated, info = self.env.step(action)

        graph = self.create_graphs(rddl_obs)  # type: ignore
        info_update, combined_graph = self._prepare_info(
            rddl_obs,  # type: ignore
            graph,
            self.add_render_graph_to_info,
        )
        info = info | info_update
        info["rddl_state"] = (
            self.env.unwrapped.state if hasattr(self.env.unwrapped, "state") else {}  # type: ignore
        )
        self.last_g = combined_graph
        self.last_rddl_obs = rddl_obs

        return graph, reward, terminated, truncated, info
