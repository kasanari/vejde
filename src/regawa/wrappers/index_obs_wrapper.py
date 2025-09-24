import logging
from typing import Any, SupportsFloat

import gymnasium as gym
from functools import cached_property
from regawa.data import HeteroObsData
from regawa.model import GroundObs
from regawa.model import BaseModel
from .graph_utils import fn_heterograph_to_heteroobs, fn_regular_map_graph_to_idx
from .stacking_utils import fn_flatten_map_graph_to_idx
from .space import HeteroStateSpace
from .types import HeteroGraph

logger = logging.getLogger(__name__)


class IndexObsWrapper(
    gym.Wrapper[
        HeteroStateSpace,
        GroundObs | tuple[int, ...],
        HeteroGraph,
        GroundObs | tuple[int, ...],
    ]
):
    """
    Converts HeteroGraph to index-based HeteroObsData
    """

    def __init__(
        self,
        env: gym.Env[HeteroGraph, GroundObs | tuple[int, ...]],
        model: BaseModel,
        stacking: bool = False,
    ) -> None:
        super().__init__(env)
        self.env = env
        self.model = model
        self._idx_to_object = ["None"]
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
        self.create_obs_dict = fn_heterograph_to_heteroobs(idx_func)

    @cached_property
    def observation_space(self) -> HeteroStateSpace:
        num_types = self.model.num_types
        num_relations = self.model.num_fluents
        max_arity = max(self.model.arity(r) for r in self.model.fluents)
        num_actions = self.model.num_actions

        return HeteroStateSpace(
            num_types,
            num_relations,
            max_arity,
            num_actions,
        )

    def step(
        self,
        action: GroundObs | tuple[int, ...],
    ) -> tuple[
        HeteroObsData,
        SupportsFloat,
        bool,
        bool,
        dict[str, Any],
    ]:
        graph, r, term, trunc, info = self.env.step(action)

        info["idx_to_object"] = graph.boolean.factors
        obs = self.create_obs_dict(graph)

        assert obs.bool.length.sum() == len(
            obs.bool.var_value
        ), "Expected {} but got {}".format(
            obs.bool.length.sum(), len(obs.bool.var_value)
        )
        assert obs.float.length.sum() == len(
            obs.float.var_value
        ), "Expected {} but got {}".format(
            obs.float.length.sum(), len(obs.float.var_value)
        )

        return obs, r, term, trunc, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[HeteroObsData, dict[str, Any]]:
        graph, info = self.env.reset(seed=seed)

        info["idx_to_object"] = graph.boolean.factors
        obs = self.create_obs_dict(graph)

        return obs, info
