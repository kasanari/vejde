from typing import Any
import gymnasium as gym
import gymnasium.spaces as spaces


class LabelingWrapper(gym.Wrapper[spaces.Dict, spaces.Dict, spaces.Tuple, spaces.Dict]):
    def __init__(self, env: gym.Env[spaces.Tuple, spaces.Dict]) -> None:
        super().__init__(env)
        self.env = env  # ActionInObsWrapper(env)

    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[str, str], float, bool, bool, dict[str, Any]]:
        (last_obs, obs), info = self.env.reset(seed=seed)
        common_keys: set[str] = set(last_obs.keys()) & set(obs.keys())
        targets = {k: v for k, v in obs.items() if k in common_keys}

        return (last_obs, obs, targets), info  # type: ignore

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[dict[str, str], float, bool, dict[str, Any]]:
        (last_obs, obs), reward, terminated, truncated, info = self.env.step(actions)

        common_keys: set[str] = set(last_obs.keys()) & set(obs.keys())

        targets = {k: v for k, v in obs.items() if k in common_keys}

        return (last_obs, obs, targets), reward, terminated, truncated, info  # type: ignore
