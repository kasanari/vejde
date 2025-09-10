import logging
from typing import Any, SupportsFloat
import gymnasium as gym
from regawa import BaseModel, GroundObs
from regawa.model.base_grounded_model import Grounding
from .graph_utils import fn_obsdict_to_graph
from .render_utils import create_render_graph, to_graphviz
from .types import HeteroGraph, RenderGraph

logger = logging.getLogger(__name__)


class GroundedGraphWrapper(
    gym.Wrapper[
        HeteroGraph, GroundObs | tuple[int, ...], GroundObs, GroundObs | tuple[int, ...]
    ]
):
    def __init__(
        self,
        env: gym.Env[GroundObs, GroundObs | tuple[int, ...]],
        model: BaseModel,
        render_mode: str = "human",
        add_render_graph_to_info: bool = True,
    ) -> None:
        super().__init__(env)
        self.model = model
        self.last_obs: dict[str, Any] = {}
        self.last_action: Grounding | None = None
        self.last_g: RenderGraph | None = None
        self._object_to_type: dict[str, str] = {"None": "None"}
        self.create_graphs = fn_obsdict_to_graph(model)

        self.add_render_graph_to_info = add_render_graph_to_info

    def render(self):
        return to_graphviz(self.last_g, scaling=10) if self.last_g is not None else None

    def _create_obs(self, rddl_observation: GroundObs) -> HeteroGraph:
        graph, _ = self.create_graphs(rddl_observation)
        return graph

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
            "rddl_state": rddl_obs,
            "action_fluents": self.model.action_fluents,
        }
        return info, combined_graph

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[HeteroGraph, dict[str, Any]]:
        super().reset(seed=seed)
        rddl_obs, info = self.env.reset(seed=seed)
        graph, _ = self.create_graphs(rddl_obs)
        info_update, combined_graph = self._prepare_info(
            rddl_obs, graph, self.add_render_graph_to_info
        )
        info = info | info_update

        self.last_g = combined_graph

        return graph, info

    def step(
        self, action: GroundObs | tuple[int, ...]
    ) -> tuple[HeteroGraph, SupportsFloat, bool, bool, dict[str, Any]]:
        rddl_obs, reward, terminated, truncated, info = self.env.step(action)

        graph, _ = self.create_graphs(rddl_obs)
        info_update, combined_graph = self._prepare_info(
            rddl_obs, graph, self.add_render_graph_to_info
        )
        info = info | info_update
        self.last_g = combined_graph
        self.last_rddl_obs = rddl_obs

        return graph, reward, terminated, truncated, info
