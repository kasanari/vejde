import itertools
import os
from typing import Any
import pyRDDLGym
import numpy as np
import gymnasium as gym
from enum import Enum


class Arity(Enum):
    CONSTANT = 0
    UNARY = 1
    BINARY = 2


def obs_to_graphviz(obs):
    # obs is a dictionary with keys "unary" and "binary"
    # "unary" is a numpy array of unary predicates
    # "binary" is a numpy array of binary predicates
    # This function returns a string that can be used to visualize the graph
    # using graphviz
    unary = obs["unary"]
    binary = obs["binary"]
    num_objects = unary.shape[0]
    graph = "digraph G {\n"
    for i in range(num_objects):
        if unary[i] == 1:
            graph += f'node{i} [label="{i}", style=filled, fillcolor=green]\n'
        else:
            graph += f'node{i} [label="{i}"]\n'
    for i in range(num_objects):
        for j in range(num_objects):
            if binary[i, j] == 1:
                graph += f"node{i} -> node{j}\n"
    graph += "}"
    return graph


def split_key(key):
    predicate, objects_str = key.split("___")
    objects = objects_str.split("__")
    return (predicate, tuple(objects))


def split_obs_keys(obs):
    return {split_key(key): value for key, value in obs.items()}


def to_graphviz(first_nodes, second_nodes, edges, edge_attributes):
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta"]
    graph = "graph G {\n"
    first_mapping = {}
    second_mapping = {}
    global_idx = 0
    for idx, data in enumerate(first_nodes):
        graph += f'"{global_idx}" [label="{data}", shape=box]\n'
        first_mapping[idx] = global_idx
        global_idx += 1
    for idx, data in enumerate(second_nodes):
        graph += f'"{global_idx}" [label="{data}", shape=circle]\n'
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


def generate_bipartite_obs(obs: dict[str, bool], symb_to_idx: dict[str, int]):
    edges: set[tuple[str, str, int]] = set()
    obs_objects: set[str] = set()

    for key in obs:
        for pos, object in enumerate(objects(key)):
            new_key = (key, object, pos)
            edges.add(new_key)
            obs_objects.add(object)

    groundings = sorted(list(obs.keys()))
    object_list: list[str] = sorted(obs_objects)

    fact_node_values = np.array([obs[key] for key in groundings], dtype=np.bool_)

    fact_node_predicate = np.array(
        [symb_to_idx[predicate(key)] for key in groundings], dtype=np.int_
    )

    edge_indices = np.array(
        [(groundings.index(key[0]), object_list.index(key[1])) for key in edges],
        dtype=np.uint,
    )

    edge_attributes = np.array([key[2] for key in edges], dtype=np.uint)

    object_nodes = np.array([symb_to_idx[object] for object in object_list])

    return (
        object_nodes,
        fact_node_values,
        fact_node_predicate,
        edge_indices,
        edge_attributes,
        groundings,
        object_list,
    )


def generate_hetero_obs(obs, object_to_index):
    new_obs = split_obs_keys(obs)

    nullary = {key: value for key, value in new_obs.items() if len(key[1]) == 0}
    unary = {key: value for key, value in new_obs.items() if len(key[1]) == 1}
    binary = {key: value for key, value in new_obs.items() if len(key[1]) == 2}

    # reshape unary to make it per object
    unary_per_object = {
        key[1][0]: {k[0]: v for k, v in unary.items() if k[1][0] == key[1][0]}
        for key, value in unary.items()
    }

    state_dict = {}
    state_dict["running"] = np.zeros(len(object_to_index), dtype=np.bool_)
    for key in obs:
        state_dict[predicate][object_idx] = int(obs[key])

    graph = {
        1: state_dict["running"],
    }

    return graph


class SysAdmin:
    def __init__(self, instance: str) -> None:
        # domain = "SysAdmin_MDP_ippc2011"
        # domain = "RecSim_ippc2023"
        domain = "Elevators_MDP_ippc2011"
        env = pyRDDLGym.make(domain, instance, enforce_action_constraints=True)
        model = env.model
        object_to_type: dict[str, str] = model.object_to_type
        types = set(object_to_type.values())
        num_objects = sum(env.model.object_counts(types))
        state_fluents: list[str] = list(env.model.state_fluents.keys())
        groundings: dict[str, list[str]] = env.model.variable_groundings
        variable_params: dict[str, list[str]] = env.model.variable_params

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
        non_fluents: list[str] = list(env.model.non_fluents.keys())
        object_to_index: dict[str, int] = model.object_to_index
        index_to_object: list[str] = list(object_to_index.keys())

        action_groundings: list[str] = list(
            dict(env.model.ground_vars_with_values(model.action_fluents)).keys()
        )
        action_edges: set[tuple[str, str, int]] = set()
        for key in action_groundings:
            object_terms = objects(key)
            for pos, object in enumerate(object_terms):
                object_to_index[object] = len(object_to_index)
                action_edges.add((key, object, pos))

        non_fluent_values: dict[str, int] = dict(
            env.model.ground_vars_with_values(model.non_fluents)
        )

        non_fluent_edges: set[tuple[str, str, int]] = set()
        for key in non_fluent_values:
            object_terms = objects(key)
            for pos, object in enumerate(object_terms):
                object_to_index[object] = len(object_to_index)
                non_fluent_edges.add((key, object, pos))

        object_terms: list[str] = list(model.object_to_index.keys())
        action_fluents: list[str] = list(model.action_fluents.keys())
        symbol_list = sorted(
            object_terms + non_fluents + state_fluents + action_fluents
        )

        self.non_fluents_values = non_fluent_values
        self.non_fluent_groundings = list(non_fluent_values.keys())
        self.non_fluent_edges = non_fluent_edges
        self.action_groundings: list[str] = action_groundings
        self.action_edges = action_edges
        self.type_to_fluent = type_to_fluent
        self.model = model
        self.num_objects = num_objects
        self.arities = arities
        self.arities_to_fluent = arities_to_fluent
        self.non_fluents = non_fluents
        self.object_to_type = object_to_type
        self.groundings = groundings
        self.state_fluents = state_fluents
        self.idx_to_symb = symbol_list
        self.symb_to_idx = {symb: idx for idx, symb in enumerate(symbol_list)}
        self.env = env
        self.observation_space = gym.spaces.Box(0, 1, (10,))
        self.action_space = gym.spaces.Discrete(len(index_to_object))

    def reset(self, seed: int | None = None):
        obs, info = self.env.reset(seed)

        (
            object_nodes,
            fact_node_values,
            fact_node_predicate,
            edge_indices,
            edge_attributes,
            groundings,
            object_list,
        ) = generate_bipartite_obs(obs, self.symb_to_idx)

        # add action fluents

        action_node_values = np.array(
            [False for _ in self.action_groundings],
            dtype=np.bool_,
        )
        action_node_symbols = np.array(
            [self.symb_to_idx[predicate(key)] for key in self.action_groundings],
            dtype=np.int_,
        )

        groundings += self.action_groundings
        fact_node_values = np.concatenate([fact_node_values, action_node_values])
        fact_node_predicate = np.concatenate([fact_node_predicate, action_node_symbols])
        edge_indices = np.concatenate(
            [
                edge_indices,
                np.array(
                    [
                        (groundings.index(s1), object_list.index(s2))
                        for s1, s2, _ in self.action_edges
                    ]
                ),
            ],
            axis=0,
        )

        edge_attributes = np.concatenate(
            [
                edge_attributes,
                np.array([v for _, _, v in self.action_edges], dtype=np.uint),
            ],
            axis=0,
        )

        # add non_fluents

        groundings += self.non_fluent_groundings

        non_fluent_nodes_values = np.array(
            [self.non_fluents_values[k] for k in self.non_fluent_groundings]
        )
        non_fluent_nodes_symbols = np.array(
            [self.symb_to_idx[predicate(k)] for k in self.non_fluent_groundings]
        )

        fact_node_values = np.concatenate(
            [fact_node_values, non_fluent_nodes_values], axis=0
        )
        fact_node_predicate = np.concatenate(
            [fact_node_predicate, non_fluent_nodes_symbols], axis=0
        )

        edge_indices = np.concatenate(
            [
                edge_indices,
                np.array(
                    [
                        (groundings.index(s1), object_list.index(s2))
                        for s1, s2, _ in self.non_fluent_edges
                    ]
                ),
            ],
            axis=0,
        )

        edge_attributes = np.concatenate(
            [
                edge_attributes,
                np.array([v for _, _, v in self.non_fluent_edges], dtype=np.uint),
            ],
            axis=0,
        )

        numeric = np.array(
            [
                1
                if self.model._variable_ranges[self.idx_to_symb[idx]] in ["real", "int"]
                else 0
                for idx in fact_node_predicate
            ],
            dtype=np.bool_,
        )

        nodes = np.stack([fact_node_values, fact_node_predicate], axis=1)

        with open("g.dot", "w") as f:
            f.write(to_graphviz(nodes, object_nodes, edge_indices, edge_attributes))
            os.system("dot -Tpng g.dot -O")

        # combine the two dictionaries
        obs = {
            "nodes": nodes,
            "numeric": numeric,
            "edge_indices": edge_indices,
            "edge_attributes": edge_attributes,
        }
        return obs, info

    def step(self, action):
        # action_name = self.index_to_action[action[0]]
        action_name = "reboot"
        object_name = self.index_to_object[action]
        rddl_action = {f"{action_name}___{object_name}": 1}
        obs, reward, terminated, truncated, info = self.env.step(rddl_action)

        graph = generate_bipartite_obs(obs, self.adjacency, self.object_to_index)

        return graph, reward, terminated, truncated, info


def main():
    instance = 1
    sysadmin = SysAdmin(instance)
    obs, info = sysadmin.reset()
    done = False
    time = 0
    sum_reward = 0
    while not done:
        time += 1
        action = (
            0,
            0,
        )  # np.random.randint(0, len(sysadmin.index_to_action)), np.random.randint(0, len(sysadmin.index_to_object))
        obs, reward, terminated, truncated, info = sysadmin.step(action)
        done = terminated or truncated
        # render = obs_to_graphviz(obs)
        # with open(f"sysadmin_{instance}_{time:0>3}.dot", "w") as f:
        # 	f.write(render)
        sum_reward += reward
        print(obs)
        print(action)
        print(reward)
    print(sum_reward)


if __name__ == "__main__":
    main()
