from copy import copy
import random
import pyRDDLGym
import numpy as np
import gymnasium as gym
from .utils import to_graphviz, to_graphviz_alt
from pyRDDLGym.core.compiler.model import RDDLLiftedModel


class RDDLGraphWrapper(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, domain: str, instance: int, render_mode: str = "human") -> None:
        env: gym.Env = pyRDDLGym.make(domain, instance, enforce_action_constraints=True)  # type: ignore
        model: RDDLLiftedModel = env.model  # type: ignore
        object_to_type: dict[str, str] = copy(model.object_to_type)  # type: ignore
        types = set(object_to_type.values())  # type: ignore
        type_list = sorted(types)
        num_objects = sum(env.model.object_counts(types))  # type: ignore
        state_fluents: list[str] = list(env.model.state_fluents.keys())  # type: ignore
        action_groundings: set[str] = set(
            dict(env.model.ground_vars_with_values(model.action_fluents)).keys()  # type: ignore
        )
        interm_fluents: set[str] = set(
            model.ground_vars_with_values(model.interm_fluents)
        )  # type: ignore

        groundings: set[str] = set(
            g
            for _, v in env.model.variable_groundings.items()  # type: ignore
            for g in v  # type: ignore
            if g[-1] != env.model.NEXT_STATE_SYM  # type: ignore
        )
        groundings = groundings - action_groundings - interm_fluents

        groundings = sorted(groundings)
        action_groundings = sorted(action_groundings)
        action_groundings = ["noop"] + action_groundings
        # groundings = ["noop"] + groundings

        variable_params: dict[str, list[str]] = copy(env.model.variable_params)  # type: ignore
        variable_params["noop"] = []
        type_to_fluent: dict[str, list[str]] = {
            value[0]: [k for k, v in variable_params.items() if v == value]
            for _, value in variable_params.items()
            if len(value) == 1
        }
        arities: dict[str, int] = {
            key: len(value) for key, value in variable_params.items()
        }
        arities_to_fluent: dict[int, list[str]] = {
            value: [k for k, v in arities.items() if v == value]
            for _, value in arities.items()
        }
        assert max(arities.values()) <= 2, "Only up to binary predicates are supported"
        non_fluents: list[str] = list(env.model.non_fluents.keys())  # type: ignore

        action_values = {k: np.bool_(False) for k in action_groundings}

        non_fluent_values: dict[str, int] = dict(
            env.model.ground_vars_with_values(model.non_fluents)  # type: ignore
        )
        action_fluents: list[str] = ["noop"] + list(model.action_fluents.keys())  # type: ignore

        relations = non_fluents + state_fluents + action_fluents

        relation_list = sorted(relations)

        object_terms: list[str] = list(model.object_to_index.keys())  # type: ignore
        object_list = sorted(object_terms)
        obj_to_idx = {symb: idx for idx, symb in enumerate(object_list)}
        # symbol_list = sorted(
        #     object_terms + non_fluents + state_fluents + action_fluents + ["noop"]
        # )
        variable_ranges: dict[str, str] = model._variable_ranges  # type: ignore
        variable_ranges["noop"] = "bool"
        # symb_to_idx = {symb: idx for idx, symb in enumerate(symbol_list)}
        num_relations = len(relation_list)
        rel_to_idx = {symb: idx for idx, symb in enumerate(relation_list)}
        num_edges = sum(arities.values())
        num_groundings = len(groundings)
        self.variable_ranges: dict[str, str] = variable_ranges
        self.non_fluents_values = non_fluent_values
        self.instance = instance
        self.domain = domain
        self.type_to_fluent = type_to_fluent
        self.model = model
        self.action_groundings = action_groundings
        self.num_types = len(type_list)
        self.num_relations = num_relations
        self.num_objects = num_objects
        self.arities = arities
        self.arities_to_fluent = arities_to_fluent
        self.non_fluents = non_fluents
        self.action_fluents = action_fluents
        self.action_values = action_values
        self.obj_to_type: dict[str, str] = object_to_type
        self.groundings = groundings
        self.state_fluents = state_fluents
        self.idx_to_obj = object_list
        self.obj_to_idx = obj_to_idx
        self.idx_to_rel = relation_list
        self.rel_to_idx = rel_to_idx
        self.type_to_idx = {symb: idx for idx, symb in enumerate(type_list)}
        self.idx_to_type = type_list

        self.env: gym.Env[gym.spaces.Dict, gym.spaces.Discrete] = env
        self.iter = 0

        # self.action_space = gym.spaces.Discrete(num_groundings)
        self.action_space = gym.spaces.MultiDiscrete(
            [
                len(action_fluents),
                num_objects,
            ]
        )

    def render(self):
        obs = self.last_obs
        nodes_classes = obs["predicate_class"]
        node_values = obs["predicate_value"]
        object_nodes = obs["object"]
        edge_indices = obs["edge_index"]
        edge_attributes = obs["edge_attr"]
        # numeric = obs["numeric"]

        with open(f"{self.domain}_{self.instance}_{self.iter}.dot", "w") as f:
            f.write(
                to_graphviz_alt(
                    nodes_classes,
                    node_values,
                    object_nodes,
                    edge_indices,
                    edge_attributes,
                    self.idx_to_type,
                    self.idx_to_rel,
                )
            )


def main():
    instance = 1
    domain = "Elevators_MDP_ippc2011"
    # domain = "SysAdmin_MDP_ippc2011"
    # domain = "RecSim_ippc2023"
    # domain = "SkillTeaching_MDP_ippc2011"
    env = RDDLGraphWrapper(domain, instance)
    obs, info = env.reset()
    env.render()
    done = False
    time = 0
    sum_reward = 0
    while not done:
        time += 1
        action = env.env.action_space.sample()

        action = random.choice(list(action.items()))
        action = {action[0]: action[1]}
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        exit()
        done = terminated or truncated

        sum_reward += reward
        # print(obs)
        print(action)
        print(reward)
    print(sum_reward)


if __name__ == "__main__":
    main()
