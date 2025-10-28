from collections.abc import Callable
import logging
from typing import Any, SupportsFloat

import gymnasium as gym
from functools import cached_property
from regawa.data import HeteroObsData
from regawa.data.graph import (
    StackedStringFactorGraph,
    StringFactorGraph,
    VariableDomain,
)
from regawa.data.obs import ObsData
from regawa.model import GroundObs
from regawa.model import BaseModel
from regawa.wrappers.utils import fn_graph_to_obsdata
from .graph_utils import fn_heterograph_to_heteroobs
from .stacking_utils import flatten_stacked_graph
from .space import HeteroStateSpace
from regawa.data import HeteroGraph

logger = logging.getLogger(__name__)


def fn_flatten_then_map_graph_to_idx(
    map_graph_to_idx: Callable[
        [StringFactorGraph[VariableDomain], type], ObsData[VariableDomain]
    ],
):
    def flatten_map_graph_to_idx(
        factorgraph: StackedStringFactorGraph[VariableDomain],
        var_val_dtype: type,
    ) -> ObsData[VariableDomain]:
        return map_graph_to_idx(
            flatten_stacked_graph(factorgraph),
            var_val_dtype,
        )

    return flatten_map_graph_to_idx


def fn_idx_obs(model: BaseModel, stacking: bool = False):
    f = fn_graph_to_obsdata(
        model.fluent_to_idx,
        model.type_to_idx,
    )

    idx_func = fn_flatten_then_map_graph_to_idx(f) if stacking else f
    create_obs_dict_fn = fn_heterograph_to_heteroobs(idx_func)

    def graph_to_obsdata(g: HeteroGraph) -> HeteroObsData:
        return create_obs_dict_fn(g)

    return graph_to_obsdata


class IndexObsWrapper(
    gym.Wrapper[
        HeteroObsData,
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
        HeteroObsData,
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
        ), "Expected {} but got {}".format(
            obs.bool.var.length.sum(), len(obs.bool.var.value)
        )
        assert obs.float.var.length.sum() == len(
            obs.float.var.value
        ), "Expected {} but got {}".format(
            obs.float.var.length.sum(), len(obs.float.var.value)
        )

        return obs, r, term, trunc, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[HeteroObsData, dict[str, Any]]:
        graph, info = self.env.reset(seed=seed)

        info["idx_to_object"] = graph.boolean.factors.names
        obs = self.create_obs_dict(graph)

        return obs, info
