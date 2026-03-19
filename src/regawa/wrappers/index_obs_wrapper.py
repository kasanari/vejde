import logging
from functools import cached_property
from typing import Any, SupportsFloat

import gymnasium as gym

from regawa.data import (
    HeteroGraph,
    HeteroIndexedFactorGraph,
    HeteroStateSpace,
    fn_idx_obs,
)
from regawa.model import BaseModel, GroundObs

logger = logging.getLogger(__name__)


class IndexObsWrapper(
    gym.Wrapper[
        HeteroIndexedFactorGraph,
        GroundObs | tuple[int, ...],
        HeteroGraph,
        GroundObs | tuple[int, ...],
    ]
):
    """
    Converts HeteroGraph to index-based HeteroIndexedFactorGraph
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
        self.create_obs_dict = fn_idx_obs(model, stacking=stacking)

    @cached_property
    def observation_space(self) -> HeteroStateSpace:  # type: ignore
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
        HeteroIndexedFactorGraph,
        SupportsFloat,
        bool,
        bool,
        dict[str, Any],
    ]:
        graph, r, term, trunc, info = self.env.step(action)

        info["idx_to_object"] = graph.boolean.factors.names
        obs = self.create_obs_dict(graph)

        assert obs.bool.var.length.sum() == len(
            obs.bool.var.value
        ), f"Expected {obs.bool.var.length.sum()} but got {len(obs.bool.var.value)}"
        assert obs.float.var.length.sum() == len(
            obs.float.var.value
        ), f"Expected {obs.float.var.length.sum()} but got {len(obs.float.var.value)}"

        return obs, r, term, trunc, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[HeteroIndexedFactorGraph, dict[str, Any]]:
        graph, info = self.env.reset(seed=seed, options=options)

        info["idx_to_object"] = graph.boolean.factors.names
        obs = self.create_obs_dict(graph)

        return obs, info
