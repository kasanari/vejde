from typing import Any, SupportsFloat
import gymnasium
from gymnasium.spaces import Dict, MultiDiscrete
import pyRDDLGym

from regawa.rddl.rddl_grounded_model import RDDLGroundedModel
from regawa.rddl.rddl_to_tuple_wrapper import RDDLToTuple
from regawa.wrappers.index_wrapper import IndexActionWrapper

from .rddl_pomdp_model import RDDLPOMDPModel
from regawa import StackingGroundedGraphWrapper, GroundedGraphWrapper
from regawa.wrappers.stacking_wrapper import StackingWrapper
from .rddl_model import RDDLModel
from ..wrappers.add_constants_wrapper import AddConstants
from .rddl_convert_enums_wrapper import RDDLConvertEnums

from rddlrepository import RDDLRepoManager
import numpy as np


from pyRDDLGym.core.parser.parser import RDDLParser
from pyRDDLGym.core.parser.reader import RDDLReader
from pyRDDLGym.core.compiler.model import RDDLLiftedModel


def make_env(
    domain, instance, model, enforce_action_constraints=False, has_enums=False
):
    env = pyRDDLGym.make(
        domain,
        instance,
        enforce_action_constraints=enforce_action_constraints,
    )  # type: ignore
    env = AddConstants(env)
    env = RDDLToTuple(env)
    if has_enums:
        env = RDDLConvertEnums(env)
    env: gymnasium.Env[Dict, MultiDiscrete] = GroundedGraphWrapper(env, model=model)
    return env


class RDDLShuffleInstancesEnv(gymnasium.Env[Dict, MultiDiscrete]):
    def __init__(
        self,
        domain: str,
        instances: list[str],
        seed: int,
        enforce_action_constraints: bool = False,
    ) -> None:
        super().__init__()

        manager = RDDLRepoManager()
        problem = manager.get_problem(domain)

        reader = RDDLReader(problem.get_domain(), problem.get_instance(instances[0]))
        parser = RDDLParser(lexer=None, verbose=False)
        parser.build()
        rddl = parser.parse(reader.rddltxt)
        model = RDDLModel(RDDLLiftedModel(rddl))

        # define the RDDL model

        self.model = model
        self.domain = domain
        self.rng = np.random.default_rng(seed)
        self.instances = instances
        self.enforce_action_constraints = enforce_action_constraints
        self.has_enums = len(self.model.model.enum_types) > 0

        # just to get the action and observation spaces
        self.env = make_env(
            domain, instances[0], model, enforce_action_constraints, self.has_enums
        )

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def step(
        self, action: MultiDiscrete
    ) -> tuple[Dict, SupportsFloat, bool, bool, dict[str, Any]]:
        return self.env.step(action)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Dict, dict[str, Any]]:
        super().reset(seed=seed)

        instance = self.rng.choice(self.instances)

        self.env = make_env(
            self.domain,
            instance,
            self.model,
            self.enforce_action_constraints,
            self.has_enums,
        )

        return self.env.reset(seed=seed, options=options)

    def render(self) -> str:
        return self.env.render()


class RDDLGraphEnv(gymnasium.Env[Dict, MultiDiscrete]):
    def __init__(
        self, domain: str, instance: str, enforce_action_constraints: bool = False
    ) -> None:
        super().__init__()
        env = pyRDDLGym.make(
            domain, instance, enforce_action_constraints=enforce_action_constraints
        )  # type: ignore
        rddl_model = env.model
        grounded_rddl_model = RDDLGroundedModel(rddl_model)
        env = AddConstants(env, grounded_rddl_model)
        model = RDDLModel(rddl_model)
        env = RDDLToTuple(env)
        if len(model.model.enum_types) > 0:
            env = RDDLConvertEnums(env)
        env: gymnasium.Env[Dict, MultiDiscrete] = GroundedGraphWrapper(env, model=model)
        env = IndexActionWrapper(env, model)
        self.env = env
        self.observation_space = env.observation_space
        self.model = model
        self.grounded_model = grounded_rddl_model

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
        grounded_model = RDDLGroundedModel(env.model)
        env = AddConstants(env, grounded_model, only_add_on_reset=True)
        env = RDDLToTuple(env)
        if len(model.model.enum_types) > 0:
            env = RDDLConvertEnums(env)
        env = StackingWrapper(env)
        env: gymnasium.Env[Dict, MultiDiscrete] = StackingGroundedGraphWrapper(
            env, model=model
        )
        env = IndexActionWrapper(env, model)
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


def register_shuffle_env():
    env_id = "RDDLShuffleInstancesEnv-v0"
    gymnasium.register(
        id=env_id,
        entry_point="regawa.rddl:RDDLShuffleInstancesEnv",
    )
    return env_id


def register_pomdp_env():
    env_id = "RDDLPOMDPGraphEnv-v0"
    gymnasium.register(
        id=env_id,
        entry_point="regawa.rddl:RDDLStackingGraphEnv",
    )
    return env_id
