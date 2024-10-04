from copy import copy
import random
from typing import Any, TypeVar, NamedTuple
import pyRDDLGym
import numpy as np
import gymnasium as gym
from enum import Enum


class Arity(Enum):
    CONSTANT = 0
    UNARY = 1
    BINARY = 2


Edge = NamedTuple("Edge", [("predicate", str), ("object", str), ("pos", int)])


def to_graphviz(
    first_nodes, second_nodes, edges, edge_attributes, idx_to_symb, numeric
):
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta"]
    graph = "graph G {\n"
    first_mapping = {}
    second_mapping = {}
    global_idx = 0
    for idx, data in enumerate(first_nodes):
        label = (
            f'"{idx_to_symb[int(data[1])]}={data[0]}"'
            if numeric[idx]
            else f'"{idx_to_symb[int(data[1])]}={bool(data[0])}"'
        )
        graph += f'"{global_idx}" [label={label}, shape=box]\n'
        first_mapping[idx] = global_idx
        global_idx += 1
    for idx, data in enumerate(second_nodes):
        graph += f'"{global_idx}" [label="{idx_to_symb[data]}", shape=circle]\n'
        second_mapping[idx] = global_idx
        global_idx += 1
    for attribute, edge in zip(edge_attributes, edges):
        graph += f'"{first_mapping[edge[0]]}" -- "{second_mapping[edge[1]]}" [color="{colors[attribute]}"]\n'
    graph += "}"
    return graph


def predicate(key: str) -> str:
    return key.split("___")[0]


def objects(key: str) -> list[str]:
    split = key.split("___")
    return split[1].split("__") if len(split) > 1 else []


def translate_edges(
    source_symbols: list[str], target_symbols: list[str], edges: set[Edge]
):
    return np.array(
        [(source_symbols.index(key[0]), target_symbols.index(key[1])) for key in edges],
        dtype=np.uint,
    )


T = TypeVar("T")


def edge_attr(edges: set[Edge]) -> np.ndarray[np.uint, Any]:
    return np.array([key[2] for key in edges], dtype=np.uint)


def create_edges(d: dict[str, Any]) -> set[Edge]:
    edges: set[Edge] = set()
    for key in d:
        for pos, object in enumerate(objects(key)):
            new_key = Edge(key, object, pos)
            edges.add(new_key)
    return edges


def generate_bipartite_obs(
    obs: dict[str, bool],
    groundings: list[str],
    symb_to_idx: dict[str, int],
    variable_ranges: dict[str, str],
) -> tuple[
    np.ndarray[np.bool_, Any],
    np.ndarray[np.int_, Any],
    np.ndarray[np.uint, Any],
    np.ndarray[np.uint, Any],
    np.ndarray[np.bool_, Any],
]:
    edges: set[Edge] = create_edges(obs)
    obs_objects: set[str] = set()

    for key in obs:
        for object in objects(key):
            obs_objects.add(object)

    object_list: list[str] = sorted(obs_objects)

    object_nodes = np.array(
        [symb_to_idx[object] for object in object_list], dtype=np.int32
    )
    fact_node_values = np.array([obs[key] for key in groundings], dtype=np.float32)

    fact_node_predicate = np.array(
        [symb_to_idx[predicate(key)] for key in groundings], dtype=np.int_
    )

    edge_indices = translate_edges(groundings, object_list, edges)

    edge_attributes = edge_attr(edges)

    fact_nodes = np.stack(
        [fact_node_values, fact_node_predicate], axis=1, dtype=np.float_
    )

    assert max(edge_indices[:, 0]) < len(fact_nodes)
    assert max(edge_indices[:, 1]) < len(object_nodes)

    numeric = np.array(
        [
            1 if variable_ranges[predicate(g)] in ["real", "int"] else 0
            for g in groundings
        ],
        dtype=np.bool_,
    )

    return (fact_nodes, object_nodes, edge_indices, edge_attributes, numeric)


class RDDLGraphWrapper(gym.Wrapper):
    metadata = {"render.modes": ["human"]}

    def __init__(self, domain: str, instance: int, render_mode: str = "human") -> None:
        env = pyRDDLGym.make(domain, instance, enforce_action_constraints=True)  # type: ignore
        model = env.model  # type: ignore
        object_to_type: dict[str, str] = copy(model.object_to_type)  # type: ignore
        types = set(object_to_type.values())  # type: ignore
        num_objects = sum(env.model.object_counts(types))  # type: ignore
        state_fluents: list[str] = list(env.model.state_fluents.keys())  # type: ignore
        action_groundings: list[str] = list(
            dict(env.model.ground_vars_with_values(model.action_fluents)).keys()  # type: ignore
        )
        groundings: list[str] = [
            g
            for _, v in env.model.variable_groundings.items()  # type: ignore
            for g in v  # type: ignore
            if g[-1] != env.model.NEXT_STATE_SYM  # type: ignore
        ]

        groundings = sorted(set(groundings))
        variable_params: dict[str, list[str]] = copy(env.model.variable_params)  # type: ignore
        self.domain = domain
        self.instance = instance
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

        action_values = {k: np.bool_(True) for k in action_groundings}
        action_edges: set[Edge] = create_edges(action_values)

        non_fluent_values: dict[str, int] = dict(
            env.model.ground_vars_with_values(model.non_fluents)  # type: ignore
        )

        non_fluent_edges: set[Edge] = create_edges(non_fluent_values)

        object_terms: list[str] = list(model.object_to_index.keys())  # type: ignore
        action_fluents: list[str] = list(model.action_fluents.keys())  # type: ignore
        symbol_list = sorted(
            object_terms + non_fluents + state_fluents + action_fluents
        )
        variable_ranges: dict[str, str] = model._variable_ranges  # type: ignore

        self.variable_ranges = variable_ranges  # type: ignore
        symb_to_idx = {symb: idx for idx, symb in enumerate(symbol_list)}
        self.non_fluents_values = non_fluent_values
        self.non_fluent_edges = non_fluent_edges
        self.action_edges = action_edges
        self.type_to_fluent = type_to_fluent
        self.model = model
        self.num_objects = num_objects
        self.arities = arities
        self.arities_to_fluent = arities_to_fluent
        self.non_fluents = non_fluents
        self.action_values = action_values
        self.object_to_type = object_to_type
        self.groundings = groundings
        self.state_fluents = state_fluents
        self.idx_to_symb = symbol_list
        self.symb_to_idx = symb_to_idx
        self.env = env
        self.iter = 0
        self.observation_space = gym.spaces.Dict(
            {
                "nodes": gym.spaces.Box(
                    low=0, high=1, shape=(len(groundings), 2), dtype=np.float32
                ),
                "object_nodes": gym.spaces.Box(
                    low=0, high=len(symbol_list), shape=(num_objects,), dtype=np.int32
                ),
                "edge_indices": gym.spaces.Box(
                    low=0,
                    high=num_objects,
                    shape=(len(action_groundings) + len(non_fluent_values), 2),
                    dtype=np.uint,
                ),
                "edge_attributes": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(len(action_groundings) + len(non_fluent_values),),
                    dtype=np.uint,
                ),
                "numeric": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(len(symbol_list),),
                    dtype=np.bool_,  # type: ignore
                ),
            }
        )
        self.action_space = gym.spaces.Discrete(len(groundings))

    def reset(self, seed: int | None = None):
        obs, info = self.env.reset(seed)

        obs |= self.action_values
        obs |= self.non_fluents_values

        observed_groundings = sorted(
            [k for k in self.groundings if isinstance(obs[k], np.bool_) and obs[k]]
        )

        filtered_obs: dict[str, Any] = {k: obs[k] for k in observed_groundings}

        (nodes, object_nodes, edge_indices, edge_attributes, numeric) = (
            generate_bipartite_obs(
                filtered_obs,
                observed_groundings,
                self.symb_to_idx,
                self.variable_ranges,
            )
        )

        # combine the two dictionaries
        obs = {
            "nodes": nodes,
            "object_nodes": object_nodes,
            "numeric": numeric,
            "edge_indices": edge_indices,
            "edge_attributes": edge_attributes,
        }

        self.iter = 0
        self.last_obs = obs

        return obs, info

    def render(self):
        obs = self.last_obs
        nodes = obs["nodes"]
        object_nodes = obs["object_nodes"]
        edge_indices = obs["edge_indices"]
        edge_attributes = obs["edge_attributes"]
        numeric = obs["numeric"]

        with open(f"{self.domain}_{self.instance}_{self.iter}.dot", "w") as f:
            f.write(
                to_graphviz(
                    nodes,
                    object_nodes,
                    edge_indices,
                    edge_attributes,
                    self.idx_to_symb,
                    numeric,
                )
            )

    def step(self, action: dict[str, int]):
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs |= self.action_values
        obs |= self.non_fluents_values

        observed_groundings = sorted(
            [k for k in self.groundings if isinstance(obs[k], np.bool_) and obs[k]]
        )

        filtered_obs: dict[str, Any] = {k: obs[k] for k in observed_groundings}

        (nodes, object_nodes, edge_indices, edge_attributes, numeric) = (
            generate_bipartite_obs(
                filtered_obs,
                observed_groundings,
                self.symb_to_idx,
                self.variable_ranges,
            )
        )

        # combine the two dictionaries
        obs = {
            "nodes": nodes,
            "object_nodes": object_nodes,
            "edge_indices": edge_indices,
            "edge_attributes": edge_attributes,
            "numeric": numeric,
        }

        self.iter += 1
        self.last_obs = obs

        return obs, reward, terminated, truncated, info


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
