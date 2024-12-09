import logging
from functools import cache
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from wrappers.stacking_wrapper import StackingWrapper

from .parent_wrapper import RDDLGraphWrapper
from .utils import predicate, to_dict_action, create_stacked_obs, get_groundings

logger = logging.getLogger(__name__)


def skip_fluent(key: str, variable_ranges: dict[str, str]) -> bool:
    return variable_ranges[predicate(key)] != "bool" or key == "noop"


class StackingGroundedRDDLGraphWrapper(RDDLGraphWrapper):
    def __init__(
        self,
        domain: str,
        instance: int,
        render_mode: str = "human",
    ) -> None:
        super().__init__(domain, instance, render_mode)
        self.env = StackingWrapper(self.env)  # type: ignore

    @property
    @cache
    def groundings(self):
        ground = super().groundings
        model = self.model
        state_fluents = model.state_fluents  # type: ignore
        observ_fluents = model.observ_fluents  # type: ignore
        # action_fluents = model.action_fluents  # type: ignore
        # action_groundings = get_groundings(model, action_fluents)  # type: ignore
        observ_groundings = get_groundings(model, observ_fluents)  # type: ignore
        state_groundings: set[str] = get_groundings(model, state_fluents)  # type: ignore
        ground = set(ground)
        ground -= state_groundings
        ground |= observ_groundings
        # ground |= action_groundings

        return sorted([g for g in ground if not skip_fluent(g, self.variable_ranges)])

    @property
    @cache
    def observation_space(self) -> spaces.Dict:  # type: ignore
        num_groundings = len(self.groundings)
        num_objects = self.num_objects
        num_types = self.num_types
        num_relations = len(self.rel_to_idx)
        num_edges = self.num_edges

        var_value_space = spaces.Sequence(spaces.Discrete(2), stack=True)

        s: dict[str, spaces.Space] = {  # type: ignore
            "var_type": spaces.Box(
                low=0,
                high=num_relations,
                shape=(num_groundings,),
                dtype=np.int64,
            ),
            "var_value": spaces.Sequence(var_value_space, stack=True),
            "factor": spaces.Box(
                low=0, high=num_types, shape=(num_objects,), dtype=np.int64
            ),
            "edge_index": spaces.Box(
                low=0,
                high=max(num_objects, num_groundings),
                shape=(num_edges, 2),
                dtype=np.int64,
            ),
            "edge_attr": spaces.Box(
                low=0,
                high=1,
                shape=(num_edges,),
                dtype=np.int64,
            ),
            "length": spaces.Box(
                low=0,
                high=40,
                shape=(num_groundings,),
                dtype=np.int64,
            ),
        }

        return spaces.Dict(s)

    def create_obs(
        self,
        rddl_obs: dict[str, list[int]],
    ):
        non_fluent_values = {k: [v] for k, v in self.non_fluent_values.items()}

        obs, g = create_stacked_obs(
            rddl_obs,
            non_fluent_values,
            self.rel_to_idx,
            self.type_to_idx,
            self.groundings,
            self.obj_to_type,
            self.variable_ranges,
            skip_fluent,
        )

        return obs, g

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[spaces.Dict, dict[str, Any]]:
        rddl_obs, info = self.env.reset(seed=seed)

        # obs |= self.action_values

        (rddl_obs, lengths) = rddl_obs

        obs, g = self.create_obs(
            rddl_obs,
        )

        obs = self._add_length(obs, lengths)

        info["state"] = g
        info["rddl_state"] = self.env.unwrapped.state  # type: ignore
        info["rddl_obs"] = rddl_obs

        self.last_obs = obs
        self.last_rddl_obs = rddl_obs

        return obs, info  # type: ignore

    def _add_length(
        self, obs: dict[str, Any], lengths: dict[str, int]
    ) -> dict[str, Any]:
        non_fluent_lengths = {k: 1 for k in self.non_fluent_values}
        lengths |= non_fluent_lengths
        length = np.array(
            [lengths[key] for key in self.groundings if key in lengths], dtype=np.int64
        )
        obs["length"] = length
        return obs

    def step(  # type: ignore
        self, action: tuple[int, int]
    ) -> tuple[spaces.Dict, SupportsFloat, bool, bool, dict[str, Any]]:
        rddl_action_dict, rddl_action = to_dict_action(
            action, self.action_fluents, self.idx_to_object, self.action_groundings
        )
        rddl_obs, reward, terminated, truncated, info = self.env.step(rddl_action_dict)
        rddl_obs, lengths = rddl_obs

        obs, g = self.create_obs(
            rddl_obs,
        )

        obs = self._add_length(obs, lengths)

        info["state"] = g
        info["rddl_state"] = self.env.unwrapped.state  # type: ignore
        info["rddl_obs"] = rddl_obs

        self.last_obs = obs
        self.last_rddl_obs = rddl_obs
        self.last_action_values = rddl_action_dict
        self.last_action = rddl_action

        return obs, reward, terminated, truncated, info  # type: ignore


def register_env():
    env_id = "StackingGroundedRDDLGraphWrapper-v0"
    gym.register(  # type: ignore
        id=env_id,
        entry_point="wrappers.pomdp_wrapper:StackingGroundedRDDLGraphWrapper",
    )
    return env_id
