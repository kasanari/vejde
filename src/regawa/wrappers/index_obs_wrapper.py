import logging
from typing import Any, SupportsFloat

import gymnasium as gym
from functools import cached_property
from regawa.data.data import HeteroObsData
from regawa.model.base_grounded_model import GroundObs
from regawa.model.base_model import BaseModel
from regawa.wrappers.graph_utils import fn_heterograph_to_heteroobs
from regawa.wrappers.space import HeteroStateSpace
from regawa.wrappers.types import HeteroGraph

logger = logging.getLogger(__name__)


class IndexObsWrapper(
    gym.Wrapper[HeteroStateSpace, GroundObs | tuple[int, ...], HeteroGraph, GroundObs | tuple[int, ...]]
):
    """
    Converts HeteroGraph to index-based HeteroObsData
    """

    def __init__(
        self, env: gym.Env[HeteroGraph, GroundObs | tuple[int, ...]], model: BaseModel
    ) -> None:
        super().__init__(env)
        self.env = env
        self.model = model
        self._idx_to_object = ["None"]
        self.create_obs_dict = fn_heterograph_to_heteroobs(model)

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

        return obs, r, term, trunc, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[HeteroObsData, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        graph, info = self.env.reset(seed=seed)

        info["idx_to_object"] = graph.boolean.factors
        obs = self.create_obs_dict(graph)

        return obs, info
