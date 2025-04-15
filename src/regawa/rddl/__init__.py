from typing import Any, SupportsFloat

import gymnasium
import numpy as np
import pyRDDLGym
from gymnasium.spaces import Dict, MultiDiscrete
from pyRDDLGym.core.compiler.model import RDDLLiftedModel
from pyRDDLGym.core.parser.parser import RDDLParser
from pyRDDLGym.core.parser.reader import RDDLReader
from rddlrepository import RDDLRepoManager

from regawa import GroundedGraphWrapper, StackingGroundedGraphWrapper
from regawa.model.base_model import BaseModel
from regawa.rddl.rddl_default_invalid_action_wrapper import RDDLDefaultInvalidActions
from regawa.wrappers import AddConstantsWrapper, IndexActionWrapper, StackingWrapper
from regawa.wrappers.remove_false_wrapper import RemoveFalseWrapper

from .rddl_convert_enums_wrapper import RDDLConvertEnums
from .rddl_grounded_model import RDDLGroundedModel
from .rddl_model import RDDLModel
from .rddl_pomdp_model import RDDLPOMDPGroundedModel
from .rddl_to_tuple_wrapper import RDDLToTuple


def model_from_domain(problem: str, instance: str) -> RDDLModel:
    reader = RDDLReader(problem, instance)
    parser = RDDLParser(lexer=None, verbose=False)
    parser.build()
    rddl = parser.parse(reader.rddltxt)
    model = RDDLModel(RDDLLiftedModel(rddl))
    return model


def make_env(
    domain: str,
    instance: str,
    has_enums: bool = False,
    remove_false: bool = False,
    stacking: bool = False,
    add_render_graph_to_info: bool = True,
):
    env = pyRDDLGym.make(domain, str(instance), enforce_action_constraints=True)  # type: ignore
    rddl_model = env.model
    model = RDDLModel(rddl_model)
    grounded_rddl_model = RDDLGroundedModel(rddl_model, remove_false=remove_false)
    env = RDDLDefaultInvalidActions(env)
    env = RDDLToTuple(env)
    env = AddConstantsWrapper(env, grounded_rddl_model, only_add_on_reset=stacking)
    env = RemoveFalseWrapper(env) if remove_false else env
    env = RDDLConvertEnums(env) if has_enums else env
    env = StackingWrapper(env) if stacking else env
    env = (
        GroundedGraphWrapper(
            env, model=model, add_render_graph_to_info=add_render_graph_to_info
        )
        if not stacking
        else StackingGroundedGraphWrapper(env, model=model)
    )
    env = IndexActionWrapper(env, model)
    return env, grounded_rddl_model, model


class RDDLCycleInstancesEnv(gymnasium.Env[Dict, MultiDiscrete]):
    def __init__(
        self,
        domain: str,
        instance: list[str],
        seed: int = 0,
        remove_false: bool = False,
        optimize: bool = False,
    ) -> None:
        super().__init__()

        manager = RDDLRepoManager()
        problem = manager.get_problem(domain)
        reader = RDDLReader(problem.get_domain(), problem.get_instance(instance[0]))
        parser = RDDLParser(lexer=None, verbose=False)
        parser.build()
        rddl = parser.parse(reader.rddltxt)
        model = RDDLModel(RDDLLiftedModel(rddl))
        current_instance = 0

        has_enums = len(model.model.enum_types) > 0
        self.domain = domain
        self.rng = np.random.default_rng(seed)
        self.instances = instance
        self.remove_false = remove_false
        self.has_enums = has_enums

        self.envs = [
            make_env(
                domain,
                str(i),
                has_enums,
                remove_false=remove_false,
                add_render_graph_to_info=(not optimize),
            )[0]
            for i in instance
        ]

        # just to get the action and observation spaces
        self.env = self.envs[current_instance]
        self.current_instance: int = current_instance
        self.model = model

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

        current_instance = (self.current_instance + 1) % len(self.instances)

        self.env = self.envs[current_instance]
        self.current_instance = current_instance

        return self.env.reset(seed=seed, options=options)

    def render(self) -> str:
        return self.env.render()


class RDDLGraphEnv(gymnasium.Env[Dict, MultiDiscrete]):
    def __init__(
        self,
        domain: str,
        instance: str,
        remove_false: bool = False,
        optimize: bool = False,
    ) -> None:
        super().__init__()
        env, grounded_model, model = make_env(
            domain,
            instance,
            remove_false=remove_false,
            add_render_graph_to_info=(not optimize),
        )
        self.env = env
        self.observation_space = env.observation_space
        self.model = model
        self.grounded_model = grounded_model

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
    def __init__(self, domain: str, instance: str, remove_false: bool) -> None:
        super().__init__()
        env, model, grounded_model = make_env(
            domain, instance, remove_false=remove_false, stacking=True
        )
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.model = model
        self.grounded_model = grounded_model

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
    env_id = "RDDLCycleInstancesEnv-v0"
    gymnasium.register(
        id=env_id,
        entry_point="regawa.rddl:RDDLCycleInstancesEnv",
    )
    return env_id


def register_pomdp_env():
    env_id = "RDDLPOMDPGraphEnv-v0"
    gymnasium.register(
        id=env_id,
        entry_point="regawa.rddl:RDDLStackingGraphEnv",
    )
    return env_id
