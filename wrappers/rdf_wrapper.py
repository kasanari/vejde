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


Edge = NamedTuple(
    "Edge",
    [
        ("subject_id", str),
        ("target_id", str),
        ("relation_id", str),
    ],
)

Graph = NamedTuple(
    "Graph",
    [
        ("nodes", np.ndarray[np.int32, Any]),
        ("edge_indices", np.ndarray[np.uint, Any]),
        ("edge_attributes", np.ndarray[np.uint, Any]),
        ("numeric", np.ndarray[np.bool_, Any]),
    ],
)


def to_graphviz(obs: dict[tuple[str, str], Graph], idx_to_symb):
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta"]
    graph = "digraph G {\n"
    value_mapping = {}
    individ_mapping = {}
    global_idx = 0

    individ_nodes = obs[("individual", "individual")].nodes
    edges = obs[("individual", "individual")].edge_indices
    edge_attributes = obs[("individual", "individual")].edge_attributes

    for idx, data in enumerate(individ_nodes):
        graph += f'"{global_idx}" [label="{idx_to_symb[data]}", shape=circle]\n'
        individ_mapping[idx] = global_idx
        global_idx += 1

    for attribute, edge in zip(edge_attributes, edges):
        graph += f'"{individ_mapping[edge[0]]}" -> "{individ_mapping[edge[1]]}" [label="{idx_to_symb[attribute]}"]\n'

    value_nodes = obs[("individual", "value")].nodes
    numeric = obs[("individual", "value")].numeric

    for idx, data in enumerate(value_nodes):
        label = f'"{data}"' if numeric[idx] else f'"{bool(data)}"'
        graph += f'"{global_idx}" [label={label}, shape=box]\n'
        value_mapping[idx] = global_idx
        global_idx += 1

    edges = obs[("individual", "value")].edge_indices
    edge_attributes = obs[("individual", "value")].edge_attributes

    for attribute, edge in zip(edge_attributes, edges):
        graph += f'"{individ_mapping[edge[0]]}" -> "{value_mapping[edge[1]]}" [label="{idx_to_symb[attribute]}"]\n'

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


def edge_attr(edges: set[Edge], symbols: dict[str, int]) -> np.ndarray[np.uint, Any]:
    # TODO edge values?
    return np.array([symbols[e[2]] for e in edges], dtype=np.uint)


def create_edges(d: dict[str, Any]) -> dict[tuple[str, str], set[Edge]]:
    edges: dict[tuple[str, str], set[Edge]] = {
        ("individual", "individual"): set(),
        ("individual", "value"): set(),
    }

    for value_id in d.keys():
        if is_fluent_to_skip(value_id, d):
            continue

        o = objects(value_id)
        p = predicate(value_id)
        if len(o) == 0:
            continue
        elif len(o) == 1:
            object = o[0]
            edges[("individual", "value")].add(Edge(object, value_id, p))
        elif len(o) == 2:
            o1, o2 = o
            edges[("individual", "individual")].add(Edge(o1, o2, p))
        else:
            raise ValueError("Only up to binary relations are supported")

    return edges


def is_false_binary_relation(key: str, obs: dict[str, bool]) -> bool:
    return isinstance(obs[key], np.bool_) and not obs[key] and len(objects(key)) == 2


def is_nullary_relation(key: str) -> bool:
    return len(objects(key)) == 0


def is_fluent_to_skip(key: str, obs: dict[str, bool]) -> bool:
    return is_false_binary_relation(key, obs) or is_nullary_relation(key)


def generate_bipartite_obs(
    obs: dict[str, bool],
    groundings: list[str],
    symb_to_idx: dict[str, int],
    variable_ranges: dict[str, str],
) -> dict[tuple[str, str], Graph]:
    edge_indices = {}
    edge_attributes = {}
    o = {}

    edges = create_edges(obs)

    obs_objects: set[str] = set()

    for key in obs:
        for object in objects(key):
            obs_objects.add(object)

    object_list: list[str] = sorted(obs_objects)

    object_nodes = np.array(
        [symb_to_idx[object] for object in object_list], dtype=np.int32
    )

    fact_node_values = np.array(
        [obs[key] for key in groundings],
        dtype=np.float32,
    )

    numeric = np.array(
        [
            1 if variable_ranges[predicate(g)] in ["real", "int"] else 0
            for g in groundings
        ],
        dtype=np.bool_,
    )

    key = ("individual", "individual")
    edge_indices[key] = translate_edges(object_list, object_list, edges[key])
    edge_attributes[key] = edge_attr(edges[key], symb_to_idx)
    o[key] = Graph(
        object_nodes,
        edge_indices[key],
        edge_attributes[key],
        np.zeros(len(object_nodes), dtype=np.bool_),
    )

    assert max(edge_indices[("individual", "individual")][:, 1]) < len(object_nodes)
    assert max(edge_indices[("individual", "individual")][:, 0]) < len(object_nodes)

    key = ("individual", "value")
    edge_indices[key] = translate_edges(object_list, groundings, edges[key])
    edge_attributes[key] = edge_attr(edges[key], symb_to_idx)
    o[key] = Graph(
        fact_node_values,
        edge_indices[key],
        edge_attributes[key],
        numeric,
    )

    assert max(edge_indices[("individual", "value")][:, 1]) < len(fact_node_values)
    assert max(edge_indices[("individual", "value")][:, 0]) < len(object_nodes)

    return o


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

        filtered_groundings = sorted(
            [g for g in self.groundings if not is_fluent_to_skip(g, obs)]
        )

        filtered_obs: dict[str, Any] = {k: obs[k] for k in filtered_groundings}

        graph = generate_bipartite_obs(
            filtered_obs,
            filtered_groundings,
            self.symb_to_idx,
            self.variable_ranges,
        )

        self.iter = 0
        self.last_obs = graph

        return graph, info

    def render(self):
        obs = self.last_obs

        with open(f"{self.domain}_{self.instance}_{self.iter}.dot", "w") as f:
            f.write(
                to_graphviz(
                    obs,
                    self.idx_to_symb,
                )
            )

    def step(self, action: dict[str, int]):
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs |= self.action_values
        obs |= self.non_fluents_values

        filtered_groundings = sorted(
            [g for g in self.groundings if not is_fluent_to_skip(g, obs)]
        )

        filtered_obs: dict[str, Any] = {k: obs[k] for k in filtered_groundings}

        graph = generate_bipartite_obs(
            filtered_obs,
            filtered_groundings,
            self.symb_to_idx,
            self.variable_ranges,
        )

        self.iter += 1
        self.last_obs = graph

        return graph, reward, terminated, truncated, info


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
