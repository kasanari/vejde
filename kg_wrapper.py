from copy import copy
import random
from typing import Any, TypeVar, NamedTuple
import pyRDDLGym
import numpy as np
import gymnasium as gym
from enum import Enum
from utils import objects, predicate
from gymnasium.spaces import MultiDiscrete, Sequence


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
        ("node_classes", np.ndarray[np.int32, Any]),
        ("edge_indices", np.ndarray[np.uint, Any]),
        ("edge_attributes", np.ndarray[np.uint, Any]),
    ],
)


def graph_to_dict(graph: Graph) -> dict[str, Any]:
    return {
        "nodes": graph.node_classes,
        "edge_index": graph.edge_indices,
        "edge_attr": graph.edge_attributes,
    }


def to_graphviz(
    obs: Graph, idx_to_obj: dict[int, str], idx_to_rel: dict[int, str]
) -> str:
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta"]
    graph = "digraph G {\n"
    value_mapping = {}
    individ_mapping = {}
    global_idx = 0

    individ_nodes = obs.nodes
    edges = obs.edge_indices
    edge_attributes = obs.edge_attributes

    for idx, data in enumerate(individ_nodes):
        graph += f'"{global_idx}" [label="{idx_to_obj[data]}", shape=circle]\n'
        individ_mapping[idx] = global_idx
        global_idx += 1

    for attribute, edge in zip(edge_attributes, edges):
        graph += f'"{individ_mapping[edge[0]]}" -> "{individ_mapping[edge[1]]}" [label="{idx_to_rel[attribute]}"]\n'

    graph += "}"
    return graph


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


def create_inverse_predicate(predicate: str) -> str:
    return f"inv({predicate})"


def create_inverse_grounding(g: str) -> str:
    o = objects(g)
    p = predicate(g)
    return f"{create_inverse_predicate(p)}___{o[1]}__{o[0]}"


def create_edges(d: dict[str, Any]) -> set[Edge]:
    edges: set[Edge] = set()

    for value_id in d.keys():
        if is_fluent_to_skip(value_id, d):
            continue

        o = objects(value_id)
        p = predicate(value_id)
        if len(o) == 0:
            continue
        elif len(o) == 1:
            object = o[0]
            edges.add(Edge(object, object, p))
        elif len(o) == 2:
            o1, o2 = o
            edges.add(Edge(o1, o2, p))
        else:
            raise ValueError("Only up to binary relations are supported")

    return edges


def is_numeric(key: str, obs: dict[str, Any]) -> bool:
    return not isinstance(obs[key], np.bool_)


def is_false_binary_relation(key: str, obs: dict[str, bool]) -> bool:
    return isinstance(obs[key], np.bool_) and not obs[key] and len(objects(key)) == 2


def is_nullary_relation(key: str) -> bool:
    return len(objects(key)) == 0


def is_fluent_to_skip(key: str, obs: dict[str, bool]) -> bool:
    return is_numeric(key, obs) or not obs[key] or is_nullary_relation(key)


def inverse_relations(groundings: set[str]) -> set[str]:
    new_groundings: set[str] = set()
    for g in groundings:
        if len(objects(g)) == 2:
            new_groundings.add(create_inverse_grounding(g))
    return new_groundings


def generate_bipartite_obs(
    obs: dict[str, bool],
    obj_to_idx: dict[str, int],
    obj_to_type_idx: dict[str, int],
    rel_to_idx: dict[str, int],
) -> Graph:
    edges = create_edges(obs)

    obs_objects: set[str] = set()

    for key in obs:
        for object in objects(key):
            obs_objects.add(object)

    object_list: list[str] = sorted(obs_objects)

    object_nodes = np.array(
        [obj_to_idx[object] for object in object_list], dtype=np.int32
    )

    object_node_classes = np.array(
        [obj_to_type_idx[object] for object in object_list], dtype=np.int32
    )

    edge_indices = translate_edges(object_list, object_list, edges)
    edge_attributes = edge_attr(edges, rel_to_idx)
    o = Graph(
        object_nodes,
        object_node_classes,
        edge_indices,
        edge_attributes,
    )

    if edge_indices.size > 0:
        assert max(edge_indices[:, 1]) < len(object_nodes)

    return o


class KGRDDLGraphWrapper(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        domain: str,
        instance: int,
        render_mode: str = "human",
        enforce_action_constraints: bool = True,
    ) -> None:
        env = pyRDDLGym.make(
            domain, instance, enforce_action_constraints=enforce_action_constraints
        )  # type: ignore
        model = env.model  # type: ignore
        object_to_type: dict[str, str] = copy(model.object_to_type)  # type: ignore
        types = set(object_to_type.values())  # type: ignore
        type_list = sorted(types)
        num_objects = sum(env.model.object_counts(types))  # type: ignore
        state_fluents: list[str] = list(env.model.state_fluents.keys())  # type: ignore
        action_groundings: set[str] = set(
            dict(env.model.ground_vars_with_values(model.action_fluents)).keys()  # type: ignore
        )
        action_groundings.add("noop")
        groundings: set[str] = (
            set(
                [
                    g
                    for _, v in env.model.variable_groundings.items()  # type: ignore
                    for g in v  # type: ignore
                    if g[-1] != env.model.NEXT_STATE_SYM  # type: ignore
                ]
            )
            - action_groundings
        )
        groundings |= inverse_relations(groundings)
        groundings = sorted(groundings)
        variable_params: dict[str, list[str]] = copy(env.model.variable_params)  # type: ignore
        variable_params["noop"] = []
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

        non_fluent_values: dict[str, int] = dict(
            env.model.ground_vars_with_values(model.non_fluents)  # type: ignore
        )

        object_terms: list[str] = list(model.object_to_index.keys())  # type: ignore
        action_fluents: list[str] = [predicate(a) for a in action_groundings]
        object_list = sorted(object_terms)

        action_mask = {
            a: {o: object_to_type[o] in variable_params[a] for o in object_terms}
            for a in action_fluents
        }

        relations = non_fluents + state_fluents
        relations += [create_inverse_predicate(r) for r in relations if arities[r] == 2]
        relation_list = sorted(relations)
        variable_ranges: dict[str, str] = model._variable_ranges  # type: ignore
        num_types = len(type_list)
        num_relations = len(relation_list)
        self.variable_ranges = variable_ranges  # type: ignore
        obj_to_idx = {symb: idx for idx, symb in enumerate(object_list)}
        rel_to_idx = {symb: idx for idx, symb in enumerate(relation_list)}
        self.non_fluents_values = non_fluent_values
        self.type_to_fluent = type_to_fluent
        self.action_fluents = action_fluents
        self.model = model
        self.num_objects = num_objects
        self.action_groundings = action_groundings
        self.num_relations = num_relations
        self.num_types = num_types
        self.num_actions = len(action_fluents)
        self.arities = arities
        self.action_mask = action_mask
        self.arities_to_fluent = arities_to_fluent
        self.non_fluents = non_fluents
        self.object_to_type = object_to_type
        self.groundings = groundings
        self.state_fluents = state_fluents
        self.idx_to_obj = object_list
        self.obj_to_idx = obj_to_idx
        self.idx_to_rel = relation_list
        self.rel_to_idx = rel_to_idx
        self.obj_to_type_idx = {
            o: type_list.index(object_to_type[o]) for o in object_list
        }
        self.env = env
        self.iter = 0
        self.observation_space = gym.spaces.Dict(
            {
                "nodes": Sequence(gym.spaces.Discrete(num_types)),
                "edge_index": Sequence(
                    gym.spaces.Box(
                        low=0,
                        high=num_objects,
                        shape=(2,),
                        dtype=np.uint,
                    )
                ),
                "edge_attr": Sequence(
                    gym.spaces.Discrete(
                        num_relations,
                    )
                ),
            }
        )
        self.action_space = MultiDiscrete([num_objects, len(action_fluents)])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict, dict[str, Any]]:
        obs, info = self.env.reset(seed)

        # obs |= self.action_values
        obs |= self.non_fluents_values

        obs |= {
            create_inverse_grounding(k): v
            for k, v in obs.items()
            if self.arities[predicate(k)] == 2
        }

        filtered_groundings = sorted(
            [g for g in self.groundings if g in obs and not is_fluent_to_skip(g, obs)]
        )

        filtered_obs: dict[str, Any] = {k: obs[k] for k in filtered_groundings}

        graph = generate_bipartite_obs(
            filtered_obs,
            self.obj_to_idx,
            self.obj_to_type_idx,
            self.rel_to_idx,
        )

        action_mask = np.stack(
            [np.array([x for x in v.values()]) for v in self.action_mask.values()]
        )

        info["action_mask"] = action_mask

        self.iter = 0
        self.last_obs = graph

        return graph_to_dict(graph), info

    def render(self):
        obs = self.last_obs

        with open(f"{self.domain}_{self.instance}_{self.iter}.dot", "w") as f:
            f.write(to_graphviz(obs, self.idx_to_obj, self.idx_to_rel))

    def step(self, action: tuple[int, int]):
        action_fluent = self.action_fluents[action[0]]
        object_id = self.idx_to_obj[action[1]]

        rddl_action = f"{action_fluent}___{object_id}"

        rddl_action_dict = (
            {rddl_action: 1} if rddl_action in self.action_groundings else {}
        )

        obs, reward, terminated, truncated, info = self.env.step(rddl_action_dict)

        obs |= self.non_fluents_values

        obs |= {
            create_inverse_grounding(k): v
            for k, v in obs.items()
            if self.arities[predicate(k)] == 2
        }

        filtered_groundings = sorted(
            [g for g in self.groundings if g in obs and not is_fluent_to_skip(g, obs)]
        )

        filtered_obs: dict[str, Any] = {k: obs[k] for k in filtered_groundings}

        action_mask = np.stack(
            [np.array([x for x in v.values()]) for v in self.action_mask.values()]
        )

        graph = generate_bipartite_obs(
            filtered_obs,
            self.obj_to_idx,
            self.obj_to_type_idx,
            self.rel_to_idx,
        )

        info["action_mask"] = action_mask

        self.iter += 1
        self.last_obs = graph

        return graph_to_dict(graph), reward, terminated, truncated, info


def main():
    instance = 1
    domain = "Elevators_MDP_ippc2011"
    # domain = "SysAdmin_MDP_ippc2011"
    # domain = "RecSim_ippc2023"
    # domain = "SkillTeaching_MDP_ippc2011"
    env = KGRDDLGraphWrapper(domain, instance)
    obs, info = env.reset(0)
    env.render()
    done = False
    time = 0
    sum_reward = 0
    while not done:
        time += 1
        action: tuple[int, int] = env.action_space.sample()
        action_fluent = action[1]
        action_mask = info["action_mask"]
        valid_objects = np.flatnonzero(action_mask[action_fluent])
        object_idx = np.random.choice(valid_objects)
        new_action = (object_idx, action_fluent)
        obs, reward, terminated, truncated, info = env.step(new_action)
        env.render()
        # exit()
        done = terminated or truncated

        sum_reward += reward
        # print(obs)
        print(new_action)
        print(reward)
    print(sum_reward)


if __name__ == "__main__":
    main()
