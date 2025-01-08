from collections.abc import Callable
import logging
from functools import cache
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gymnasium.spaces import Discrete, Sequence, Dict, MultiDiscrete

from .utils import predicate, to_dict_action, create_obs

from .utils import to_graphviz_alt, to_graphviz

from model.base_model import BaseModel

logger = logging.getLogger(__name__)


def skip_fluent(key: str, variable_ranges: dict[str, str]) -> bool:
    return False


class GroundedRDDLGraphWrapper(gym.Wrapper[dict[str, Any], MultiDiscrete, Dict, Dict]):
    @property
    def metadata(self) -> dict[str, Any]:
        return {"render_modes": ["human", "idx"]}

    @metadata.setter
    def metadata(self, value: dict[str, Any]):
        self._metadata = value

    def __init__(
        self,
        env: gym.Env[dict[str, Any], dict[str, int]],
        model: BaseModel,
        render_mode: str = "human",
    ) -> None:
        super().__init__(env)
        self.wrapped_model = model
        self.env = env
        self.last_obs: dict[str, Any] = {}
        self.iter = 0
        self._idx_to_object: list[str] = []
        self._idx_to_object_type: list[str] = []

    def idx_to_object(self, idx: int) -> str:
        return self._idx_to_object[idx]

    def idx_to_object_type(self, idx: int) -> str:
        return self._idx_to_object_type[idx]

    @property
    def action_space(self) -> gym.spaces.MultiDiscrete:  # type: ignore
        return gym.spaces.MultiDiscrete(
            [
                self.wrapped_model.num_actions,  # type: ignore
                len(self._idx_to_object) or 1,
            ]
        )

    def render(self):
        return to_graphviz(self.last_g)

        if self.metadata["render_modes"] == "idx":
            obs = self.last_obs
            nodes_classes = obs["var_type"]
            node_values = obs["var_value"]
            object_nodes = obs["factor"]
            edge_indices = obs["edge_index"].T
            edge_attributes = obs["edge_attr"]
            # numeric = obs["numeric"]

            return to_graphviz_alt(
                nodes_classes,
                node_values,
                object_nodes,
                edge_indices,  # type: ignore
                edge_attributes,
                self.wrapped_model.idx_to_type,  # type: ignore
                self.wrapped_model.idx_to_fluent,  # type: ignore
            )

    @property
    @cache
    def observation_space(self) -> spaces.Dict:  # type: ignore
        # num_groundings = len(self.wrapped_model.groundings)
        # num_objects = self.wrapped_model.num_objects
        num_types = self.wrapped_model.num_types
        num_relations = self.wrapped_model.num_fluents

        s: dict[str, spaces.Space] = {  # type: ignore
            "var_type": Sequence(Discrete(num_relations), stack=True),
            "var_value": Sequence(
                Discrete(
                    2,
                ),
                stack=True,
            ),
            "factor": Sequence(
                Discrete(
                    num_types,
                ),
                stack=True,
            ),
            "edge_index": Sequence(
                spaces.Box(
                    low=0,
                    high=2000,
                    shape=(2,),
                    dtype=np.int64,
                ),
                stack=True,
            ),
            "edge_attr": Sequence(Discrete(2), stack=True),
            "length": Sequence(Discrete(1), stack=True),
            "n_nodes": Discrete(2000),
        }

        return spaces.Dict(s)

    def _create_obs(
        self, rddl_obs: dict[str, list[int]]
    ) -> tuple[spaces.Dict, dict[str, Any]]:
        o, g, _ = create_obs(
            rddl_obs,
            self.wrapped_model,
            skip_fluent,
        )

        o["length"] = np.ones_like(o["var_value"])
        o["n_nodes"] = len(g.factors)
        return o, g

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[spaces.Dict, dict[str, Any]]:
        rddl_obs, info = self.env.reset(seed=seed)

        obs, g = self._create_obs(rddl_obs)

        info["state"] = g
        info["rddl_state"] = rddl_obs  # type: ignore

        self.last_obs = obs
        self.last_g = g
        self.last_rddl_obs = rddl_obs
        self._idx_to_object_type = g.factor_values
        self._idx_to_object = g.factors

        return obs, info

    def _to_rddl_action(
        self, action: spaces.MultiDiscrete
    ) -> tuple[dict[str, int], str]:
        return to_dict_action(
            action,
            self.wrapped_model.idx_to_action,
            self.idx_to_object_type,
            self.idx_to_object,
            self.wrapped_model.fluent_params,
        )

    def step(
        self, action: spaces.MultiDiscrete
    ) -> tuple[spaces.Dict, SupportsFloat, bool, bool, dict[str, Any]]:
        rddl_action, grounded_action = self._to_rddl_action(
            action,
        )
        rddl_obs, reward, terminated, truncated, info = self.env.step(rddl_action)

        obs, g = self._create_obs(rddl_obs)

        info["state"] = g
        info["rddl_state"] = rddl_obs  # type: ignore
        info["rddl_action"] = rddl_action

        self.last_obs = obs
        self.last_g = g
        self.last_rddl_obs = rddl_obs
        self.last_action = grounded_action
        self._idx_to_object_type = g.factor_values  # type: ignore
        self._idx_to_object = g.factors  # type: ignore

        return obs, reward, terminated, truncated, info
