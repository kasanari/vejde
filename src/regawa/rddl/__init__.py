from typing import Any, SupportsFloat
import gymnasium
from gymnasium.spaces import Dict, MultiDiscrete
import pyRDDLGym

from regawa.rddl.rddl_to_tuple_wrapper import RDDLToTuple

from .rddl_pomdp_model import RDDLPOMDPModel
from regawa import StackingGroundedGraphWrapper, GroundedGraphWrapper
from regawa.wrappers.stacking_wrapper import StackingWrapper
from .rddl_model import RDDLModel
from .rddl_add_non_fluents_wrapper import RDDLAddNonFluents
from .rddl_convert_enums_wrapper import RDDLConvertEnums


class RDDLGraphEnv(gymnasium.Env[Dict, MultiDiscrete]):
    def __init__(
        self, domain: str, instance: str, enforce_action_constraints: bool = False
    ) -> None:
        super().__init__()
        env = pyRDDLGym.make(
            domain, instance, enforce_action_constraints=enforce_action_constraints
        )  # type: ignore
        model = RDDLModel(env.model)
        env = RDDLAddNonFluents(env)
        env = RDDLToTuple(env)
        if len(model.model.enum_types) > 0:
            env = RDDLConvertEnums(env)
        env: gymnasium.Env[Dict, MultiDiscrete] = GroundedGraphWrapper(env, model=model)
        self.env = env
        self.observation_space = env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def step(
        self, action: MultiDiscrete
    ) -> tuple[Dict, SupportsFloat, bool, bool, dict[str, Any]]:
        return self.env.step(action)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Dict, dict[str, Any]]:
        super().reset(seed=seed)
        return self.env.reset(seed=seed, options=options)

    def render(self) -> str:
        return self.env.render()


class RDDLStackingGraphEnv(gymnasium.Env[Dict, MultiDiscrete]):
    def __init__(
        self, domain: str, instance: str, enforce_action_constraints: bool = False
    ) -> None:
        super().__init__()
        env = pyRDDLGym.make(
            domain, instance, enforce_action_constraints=enforce_action_constraints
        )  # type: ignore
        model = RDDLPOMDPModel(env.model)
        env = RDDLAddNonFluents(env, only_add_on_reset=True)
        env = RDDLToTuple(env)
        if len(model.model.enum_types) > 0:
            env = RDDLConvertEnums(env)
        env = StackingWrapper(env)
        env: gymnasium.Env[Dict, MultiDiscrete] = StackingGroundedGraphWrapper(
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
        super().reset(seed=seed)
        return self.env.reset(seed=seed, options=options)

    def render(self) -> str:
        return self.env.render()


def register_env():
    env_id = "RDDLGraphEnv-v0"
    gymnasium.register(
        id=env_id,
        entry_point="regawa.rddl:RDDLGraphEnv",
    )
    return env_id


def register_pomdp_env():
    env_id = "RDDLPOMDPGraphEnv-v0"
    gymnasium.register(
        id=env_id,
        entry_point="regawa.rddl:RDDLStackingGraphEnv",
    )
    return env_id
