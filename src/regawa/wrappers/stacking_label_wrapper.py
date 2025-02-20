from typing import Any

import gymnasium as gym
import gymnasium.spaces as spaces


class LabelingWrapper(gym.Wrapper[spaces.Dict, spaces.Dict, spaces.Tuple, spaces.Dict]):
    def __init__(self, env: gym.Env[spaces.Tuple, spaces.Dict]) -> None:
        super().__init__(env)
        self.env = env  # ActionInObsWrapper(env)
        self.observation_space = spaces.Dict(
            {
                "obs": env.observation_space,
                "targets": spaces.Sequence(Text(10)),
            }
        )

    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[str, str], float, bool, bool, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed)
        return obs, info  # type: ignore

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[dict[str, str], float, bool, dict[str, Any]]:
        """""
        Stacks observations
        obs= {
            "key1" [t1, t2, t3, t4],
            "key2" [t1, t2, t3, t4],
            "key3" [None, None, t3, t4],
        }
        """ ""
        # (obs_with_actions, next_obs), reward, terminated, truncated, info = (
        #     self.env.step(actions)
        # )

        (last_obs, obs), reward, terminated, truncated, info = self.env.step(actions)

        common_keys: set[str] = set(last_obs.keys()) & set(obs.keys())
        temporal_edges: list[tuple[str, str]] = [
            (k, k) for k in common_keys if obs[k] is not None or last_obs[k] is not None
        ]

        inputs = {k: last_obs[k] for k in last_obs if k in common_keys}
        targets = {k: v for k, v in obs.items() if k in common_keys}

        return obs, reward, terminated, truncated, info  # type: ignore
