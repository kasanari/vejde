from typing import Any, SupportsFloat
import gymnasium
from gymnasium.spaces import Dict, MultiDiscrete
import pyRDDLGym
from .rddl_model import RDDLModel
from .rddl_add_non_fluents_wrapper import RDDLAddNonFluents
from wrappers.wrapper import GroundedRDDLGraphWrapper


class RDDLGraphEnv(gymnasium.Env[Dict, MultiDiscrete]):
    def __init__(self, domain: str, instance: str) -> None:
        super().__init__()
        env = pyRDDLGym.make(domain, instance, enforce_action_constraints=True)  # type: ignore
        model = RDDLModel(env.model)
        env = RDDLAddNonFluents(env)
        env: gymnasium.Env[Dict, MultiDiscrete] = GroundedRDDLGraphWrapper(
            env, model=model
        )
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(
        self, action: MultiDiscrete
    ) -> tuple[Dict, SupportsFloat, bool, bool, dict[str, Any]]:
        return self.env.step(action)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Dict, dict[str, Any]]:
        return self.env.reset(seed=seed, options=options)


def register_env():
    env_id = "RDDLGraphEnv-v0"
    gymnasium.register(
        id=env_id,
        entry_point="rddl:RDDLGraphEnv",
    )
    return env_id
