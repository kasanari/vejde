from typing import Any
from pyRDDLGym.core.compiler.model import RDDLLiftedModel  # type: ignore
import pyRDDLGym  # type: ignore

from wrappers.utils import get_groundings


def add_actions_to_obs(
    obs: dict[str, Any], actions: dict[str, bool | None]
) -> dict[str, Any]:
    obs_with_actions = {a: bool(v) for a, v in actions.items()} | obs
    return obs_with_actions


class ActionInObsWrapper:
    """
    Adds actions to the previous observation.
    """

    def __init__(self, env: pyRDDLGym.RDDLEnv) -> None:
        self.env = env
        self.last_obs: dict[str, Any] = {}

    @staticmethod
    def _add_actions_to_obs(
        model: RDDLLiftedModel, obs: dict[str, Any], actions: dict[str, int]
    ) -> dict[str, bool | None]:
        action_groundings = get_groundings(model, model.action_fluents)  # type: ignore

        boolean_actions: dict[str, bool] = {k: bool(v) for k, v in actions.items()}

        new_actions: dict[str, bool | None] = {
            k: boolean_actions.get(k, None) for k in action_groundings if k not in obs
        }

        obs_with_actions = add_actions_to_obs(obs, new_actions)
        return obs_with_actions

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[
        tuple[dict[str, bool | None], dict[str, bool | None]],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        next_obs, reward, terminated, truncated, info = self.env.step(actions)

        obs_with_actions = self._add_actions_to_obs(
            self.env.model, self.last_obs.copy(), actions
        )

        self.last_obs = next_obs

        return (obs_with_actions, next_obs), reward, terminated, truncated, info

    def reset(
        self, seed: int | None = None
    ) -> tuple[
        dict[str, bool | None],
        dict[str, Any],
    ]:
        obs, info = self.env.reset(seed=seed)

        obs_with_actions = self._add_actions_to_obs(self.env.model, obs, {})

        self.last_obs = obs
        return obs_with_actions, info

    @property
    def unwrapped(self) -> pyRDDLGym.RDDLEnv:
        return self.env.unwrapped  # type: ignore
