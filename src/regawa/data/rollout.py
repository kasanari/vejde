from collections import deque

import numpy as np
from numpy.typing import NDArray
from regawa.data.buffer import HeteroGraphBuffer


from typing import NamedTuple

from regawa.data.obs import HeteroObsData
import json


class Rollout(NamedTuple):
    rewards: list[float]
    obs: HeteroGraphBuffer
    actions: list[tuple[int, ...]]
    values: list[float]


class Serializer(json.JSONEncoder):
    def default(self, o: object):
        if isinstance(o, NDArray):  # type: ignore
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)  # type: ignore
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_rollout(rollout: Rollout, path: str):
    with open(path, "w") as f:
        json.dump(rollout._asdict(), f, cls=Serializer)


def load_rollout(path: str) -> Rollout:
    with open(path, "r") as f:
        data = json.load(f)
    return Rollout(**data)


class RolloutCollector:
    rewards: deque[float]
    obs: HeteroGraphBuffer
    actions: deque[tuple[int, ...]]

    def __init__(self) -> None:
        self.rewards = deque()
        self.obs = HeteroGraphBuffer()
        self.actions = deque()

    def add_single(
        self, obs: HeteroObsData, action: tuple[int, ...], reward: float
    ) -> None:
        self.rewards.append(reward)
        self.obs.add_single_dict(obs)
        self.actions.append(action)

    def export(self) -> Rollout:
        return Rollout(
            rewards=list(self.rewards),
            obs=self.obs,
            actions=list(self.actions),
            values=self.values,
        )

    @property
    def return_(self) -> float:
        return sum(self.rewards)

    @property
    def values(self) -> list[float]:
        returns = [sum(list(self.rewards)[i:]) for i in range(len(self.rewards))]
        return returns
