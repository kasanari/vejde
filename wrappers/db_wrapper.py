from copy import copy
import random
from typing import Any, Callable, TypeVar, NamedTuple
import pyRDDLGym
import numpy as np
import gymnasium as gym
from enum import Enum

from pyRDDLGym.core.env import RDDLEnv
from wrappers.utils import objects, predicate, arity
from gymnasium.spaces import MultiDiscrete, Sequence
from pyRDDLGym.core.compiler.model import RDDLLiftedModel
import logging

logger = logging.getLogger(__name__)


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

IdxGraph = NamedTuple(
    "IdxGraph",
    [
        ("nodes", np.ndarray[np.int64, Any]),
        ("node_classes", np.ndarray[np.int64, Any]),
        ("edge_indices", np.ndarray[np.int64, Any]),
        ("edge_attributes", np.ndarray[np.int64, Any]),
    ],
)

Graph = NamedTuple(
    "Graph",
    [
        ("nodes", list[str]),
        ("node_classes", list[str]),
        ("edge_indices", np.ndarray[np.int64, Any]),
        ("edge_attributes", list[str]),
    ],
)


def graph_to_dict(graph: IdxGraph, types_instead_of_objects: bool) -> dict[str, Any]:
    return {
        "nodes": graph.node_classes if types_instead_of_objects else graph.nodes,
        "edge_index": graph.edge_indices,
        "edge_attr": graph.edge_attributes,
    }


def to_graphviz(
    obs: IdxGraph,
    idx_to_obj: list[str],
    idx_to_rel: list[str],
    object_to_type: dict[str, str],
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
        obj = idx_to_obj[data]
        obj_type = object_to_type[obj]
        graph += (
            f'"{global_idx}" [label="{obj_type}({idx_to_obj[data]})", shape=circle]\n'
        )
        individ_mapping[idx] = global_idx
        global_idx += 1

    for attribute, edge in zip(edge_attributes, edges):
        graph += f'"{individ_mapping[edge[0]]}" -> "{individ_mapping[edge[1]]}" [label="{idx_to_rel[attribute]}"]\n'

    graph += "}"
    return graph


def to_graphviz_dict(
    obs: dict[str, list[int]],
) -> str:
    graph = "digraph G {\n"
    individ_mapping = {}
    global_idx = 0

    individ_nodes = obs["nodes"]
    edges = obs["edge_index"]
    edge_attributes = obs["edge_attr"]

    for idx, data in enumerate(individ_nodes):
        graph += f'"{global_idx}" [label="{data}", shape=circle]\n'
        individ_mapping[idx] = global_idx
        global_idx += 1

    for attribute, edge in zip(edge_attributes, edges):
        graph += f'"{individ_mapping[edge[0]]}" -> "{individ_mapping[edge[1]]}" [label="{attribute}"]\n'

    graph += "}"
    return graph


def translate_edges(
    source_symbols: list[str], target_symbols: list[str], edges: set[Edge]
):
    return np.array(
        [(source_symbols.index(key[0]), target_symbols.index(key[1])) for key in edges],
        dtype=np.int64,
    )


T = TypeVar("T")


def edge_attr(edges: set[Edge]) -> list[str]:
    # TODO edge values?
    return [e[2] for e in edges]


def create_inverse_predicate(predicate: str) -> str:
    return f"inv({predicate})"


def create_inverse_grounding(g: str) -> str:
    o = objects(g)
    p = predicate(g)
    return f"{create_inverse_predicate(p)}___{o[1]}__{o[0]}"


def create_edges(d: dict[str, Any]) -> set[Edge]:
    edges: list[Edge] = []

    for value_id in d.keys():
        o = objects(value_id)
        p = predicate(value_id)
        if len(o) == 0:
            continue
        elif len(o) == 1:
            object = o[0]
            edges.append(Edge(object, object, p))
        elif len(o) == 2:
            o1, o2 = o
            edges.append(Edge(o1, o2, p))
        else:
            raise ValueError("Only up to binary relations are supported")

    return edges


def is_numeric(key: str, obs: dict[str, Any]) -> bool:
    return not isinstance(obs[key], np.bool_)


def is_false_binary_relation(key: str, obs: dict[str, bool]) -> bool:
    return isinstance(obs[key], np.bool_) and not obs[key] and arity(key) == 2


def is_nullary_relation(key: str) -> bool:
    return len(objects(key)) == 0


def skip_grounding(
    key: str, variable_ranges: dict[str, str], arities: dict[str, int]
) -> bool:
    return skip_relation(predicate(key), variable_ranges, arities, True)


def skip_grounding_with_val(
    key: str, variable_ranges: dict[str, str], arities: dict[str, int], value: bool
) -> bool:
    return skip_relation(predicate(key), variable_ranges, arities, value)


def skip_relation(
    key: str,
    variable_ranges: dict[str, str],
    arities: dict[str, int],
    value: bool,
) -> bool:
    if arities[key] == 0:
        return True

    if arities[key] == 1:
        return variable_ranges[key] != "bool"

    if arities[key] == 2:
        return variable_ranges[key] != "bool" or (not value)

    return False


def inverse_relations(groundings: set[str]) -> set[str]:
    new_groundings: set[str] = set()
    for g in groundings:
        if len(objects(g)) == 2:
            new_groundings.add(create_inverse_grounding(g))
    return new_groundings


class Table:
    def __init__(self, header: list[str], types: list[type]) -> None:
        self.header = tuple(header)
        self.types = tuple(types)
        self.tuples = set()

    def add_tuple(self, t: tuple) -> None:
        self.tuples.add(t)

    def __repr__(self) -> str:
        header_type = zip(self.header, self.types)
        header = ", ".join([f"{h}: {t}" for h, t in header_type])
        rep = header
        for t in self.tuples:
            rep += "\n"
            rep += ", ".join(map(str, t))
        return rep


def db_to_indices(
    objects: dict[str, dict[str, Any]],
    associations: dict[tuple[str, str, str], dict[tuple[str, str], Any]],
    obj_to_idx: Callable[[str], int],
    type_to_idx: dict[str, int],
    rel_to_idx: dict[str, int],
):
    nodes = {
        t: np.stack([np.array(list(tup.values())) for tup in v.values()])
        for t, v in objects.items()
    }

    def map_edge(e: tuple[str, str, str]) -> tuple[int, int, int]:
        return type_to_idx[e[0]], rel_to_idx[e[1]], type_to_idx[e[2]]

    edge_indices = {
        str(k): np.stack(
            [np.array([obj_to_idx(o1), obj_to_idx(o2)]) for o1, o2 in v.keys()]
        )
        for k, v in associations.items()
    }

    edge_attr = {
        str(k): np.array([rel_to_idx[k[1]] for _ in v.keys()])
        for k, v in associations.items()
    }

    return {
        "nodes": nodes,
        "edge_index": edge_indices,
        # "edge_attr": edge_attr,
    }


def message_type(g: str, obj_to_type: dict[str, str]) -> tuple[str, str, str]:
    o = objects(g)
    p = predicate(g)
    o1, o2 = o
    return (obj_to_type[o1], p, obj_to_type[o2])


def obs_to_db(dict_obs: dict[str, Any], obj_to_type: dict[str, str]) -> dict[str, bool]:
    per_object: dict[str, dict[str, Any]] = {}

    per_type: dict[str, list] = {}

    associations: dict[tuple, list] = {}

    # create 'tables'
    for value_id in dict_obs.keys():
        o = objects(value_id)
        p = predicate(value_id)
        if len(o) == 0:
            continue
        elif len(o) == 1:
            (object,) = o
            per_object[object] = {}
            per_type[obj_to_type[object]] = {}
        elif len(o) == 2:
            o1, o2 = o
            associations[(obj_to_type[o1], p, obj_to_type[o2])] = {}
        else:
            raise ValueError("Only up to binary relations are supported")

    for value_id in dict_obs.keys():
        o = objects(value_id)
        p = predicate(value_id)
        if len(o) == 0:
            continue
        elif len(o) == 1:
            (object,) = o
            per_object[object][p] = dict_obs[value_id]
        elif len(o) == 2:
            o1, o2 = o
            associations[(obj_to_type[o1], p, obj_to_type[o2])][(o1, o2)] = dict_obs[
                value_id
            ]
        else:
            raise ValueError("Only up to binary relations are supported")

    for object, data in per_object.items():
        per_type[obj_to_type[object]][object] = data

    return per_type, associations


def map_to_indices(
    graph: Graph,
    obj_to_idx: dict[str, int],
    type_to_idx: dict[str, int],
    rel_to_idx: dict[str, int],
) -> IdxGraph:
    object_list = graph.nodes
    types = graph.node_classes
    edge_attr = graph.edge_attributes
    edge_indices = graph.edge_indices

    object_nodes = np.array(
        [obj_to_idx[object] for object in object_list], dtype=np.int64
    )

    object_node_classes = np.array([type_to_idx[t] for t in types], dtype=np.int64)

    edge_attr_idx = np.array([rel_to_idx[e] for e in edge_attr], dtype=np.int64)

    if edge_indices.size > 0:
        assert max(edge_indices[:, 1]) < len(object_nodes)

    return IdxGraph(object_nodes, object_node_classes, edge_indices, edge_attr_idx)


def generate_bipartite_obs(
    obs: dict[str, bool],
    obj_to_type: dict[str, str],
) -> Graph:
    edges = create_edges(obs)

    obs_objects: set[str] = set()

    for key in obs:
        for object in objects(key):
            obs_objects.add(object)

    object_list: list[str] = sorted(obs_objects)

    object_class = [obj_to_type[object] for object in object_list]

    edge_attributes = edge_attr(edges)

    edge_indices = translate_edges(object_list, object_list, edges)

    o = Graph(
        object_list,
        object_class,
        edge_indices,
        edge_attributes,
    )

    return o


class KGRDDLGraphWrapper(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        domain: str,
        instance: int,
        render_mode: str = "human",
        enforce_action_constraints: bool = True,
        add_inverse_relations: bool = True,
        types_instead_of_objects: bool = False,
    ) -> None:
        env: RDDLEnv = pyRDDLGym.make(
            domain, instance, enforce_action_constraints=enforce_action_constraints
        )
        model: RDDLLiftedModel = env.model  # type: ignore
        variable_params: dict[str, list[str]] = copy(model.variable_params)  # type: ignore
        object_to_type = copy(model.object_to_type)  # type: ignore
        variable_ranges: dict[str, str] = copy(model._variable_ranges)  # type: ignore
        object_to_type: dict[str, str]
        types = set(object_to_type.values())  # type: ignore
        type_list = sorted(types)
        num_objects = sum(model.object_counts(types))  # type: ignore
        state_fluents: list[str] = list(model.state_fluents.keys())  # type: ignore
        action_groundings: set[str] = set(
            dict(model.ground_vars_with_values(model.action_fluents)).keys()  # type: ignore
        )

        interm_fluents: set[str] = set(
            model.ground_vars_with_values(model.interm_fluents)
        )  # type: ignore

        object_per_type: dict[str, list[str]] = {
            t: [o for o, ot in object_to_type.items() if ot == t]  # type: ignore
            for t in type_list
        }

        action_groundings.add("noop")
        arities: dict[str, int] = {
            key: len(value) for key, value in variable_params.items()
        }
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
            - interm_fluents
        )

        groundings = {
            g
            for g in groundings
            if not skip_grounding(
                g,
                variable_ranges,
                arities,
            )
        }  # type: ignore

        message_types = {
            message_type(g, object_to_type) for g in groundings if arity(g) == 2
        }

        if add_inverse_relations:
            groundings |= inverse_relations(groundings)
        groundings = sorted(groundings)

        predicates_per_type: dict[str, set[str]] = {t: set() for t in type_list}

        for g in groundings:
            p = predicate(g)
            if arity(g) != 1:  # TODO
                continue
            for o in objects(g):
                t = object_to_type[o]
                predicates_per_type[t].add(p)

        variable_params["noop"] = []
        self.domain = domain
        self.instance = instance
        type_to_fluent: dict[str, list[str]] = {
            value[0]: [k for k, v in variable_params.items() if v == value]
            for _, value in variable_params.items()
            if len(value) == 1
        }

        arities_to_fluent: dict[int, list[str]] = {
            value: [k for k, v in arities.items() if v == value]
            for _, value in arities.items()
        }
        assert max(arities.values()) <= 2, "Only up to binary predicates are supported"
        non_fluents: list[str] = list(model.non_fluents.keys())

        non_fluent_values: dict[str, int] = dict(
            model.ground_vars_with_values(model.non_fluents)  # type: ignore
        )

        object_terms: list[str] = list(model.object_to_index.keys())  # type: ignore
        action_fluents: list[str] = sorted(set(predicate(a) for a in action_groundings))

        action_mask = {
            a: {
                str(o): str(object_to_type[o]) in variable_params[a]
                for o in object_terms
            }
            for a in action_fluents
        }

        relations = non_fluents + state_fluents
        if add_inverse_relations:
            relations |= set(
                create_inverse_predicate(r) for r in relations if arities[r] == 2
            )

        relations = {
            str(r)
            for r in relations
            if not skip_relation(r, variable_ranges, arities, True)
        }

        num_predicates_per_type = {k: len(v) for k, v in predicates_per_type.items()}

        relation_list = sorted(relations)
        num_types = len(type_list)
        num_relations = len(relation_list)
        obj_to_idx = {
            t: {symb: idx for idx, symb in enumerate(o)}
            for t, o in object_per_type.items()
        }
        rel_to_idx = {symb: idx for idx, symb in enumerate(relation_list)}
        self.variable_ranges: dict[str, str] = variable_ranges
        self.non_fluents_values = non_fluent_values
        self.type_to_fluent = type_to_fluent
        self.num_predicates_per_type = num_predicates_per_type
        self.action_fluents = action_fluents
        self.model: RDDLLiftedModel = model
        self.num_objects = num_objects
        self.action_groundings = action_groundings
        self.num_relations = num_relations
        self.num_types = num_types
        self.num_actions = len(action_fluents)
        self.arities = arities
        self.action_mask = action_mask
        self.arities_to_fluent = arities_to_fluent
        self.non_fluents = non_fluents
        self.object_to_type: dict[str, str] = object_to_type
        self.groundings = groundings
        self.state_fluents = state_fluents
        self._obj_to_idx = obj_to_idx
        self.idx_to_rel = relation_list
        self.rel_to_idx = rel_to_idx
        self.add_inverse_relations = add_inverse_relations
        self.types_instead_of_objects = types_instead_of_objects
        self.type_to_idx = {t: idx for idx, t in enumerate(type_list)}
        self.idx_to_type = type_list
        self.env: RDDLEnv = env
        self.iter = 0

        node_spaces = gym.spaces.Dict(
            {
                k: Sequence(
                    gym.spaces.Box(
                        low=0,
                        high=1,
                        shape=(num_predicates_per_type[k],),
                    ),
                    stack=True,
                )
                for k in type_list
            }
        )

        edge_index_spaces = gym.spaces.Dict(
            {
                str(m): Sequence(
                    gym.spaces.Box(
                        low=0,
                        high=max(
                            len(object_per_type[m[0]]), len(object_per_type[m[2]])
                        ),
                        shape=(2,),
                        dtype=np.int64,
                    ),
                    stack=True,
                )
                for m in message_types
            }
        )

        self.observation_space = gym.spaces.Dict(
            {
                "nodes": node_spaces,
                "edge_index": edge_index_spaces,
            }
        )

        self.action_space = MultiDiscrete(
            [
                len(action_fluents),
                num_objects,
            ]
        )

    def object_to_index(self, obj: str) -> int:
        return self._obj_to_idx[self.object_to_type[obj]][obj]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, info = self.env.reset(seed)

        # obs |= self.action_values
        obs |= self.non_fluents_values

        if self.add_inverse_relations:
            obs |= {
                create_inverse_grounding(k): v
                for k, v in obs.items()
                if self.arities[predicate(k)] == 2
            }

        filtered_groundings = sorted(
            [
                g
                for g in self.groundings
                if g in obs
                and not skip_grounding_with_val(
                    g, self.variable_ranges, self.arities, obs[g]
                )
            ]
        )

        filtered_obs: dict[str, Any] = {k: obs[k] for k in filtered_groundings}

        tabs, assocs = obs_to_db(filtered_obs, self.object_to_type)
        o = db_to_indices(
            tabs,
            assocs,
            self.object_to_index,
            self.type_to_idx,
            self.rel_to_idx,
        )

        action_mask = np.stack(
            [np.array([x for x in v.values()]) for v in self.action_mask.values()]
        )

        info["action_mask"] = action_mask
        info["state"] = (tabs, assocs)

        self.iter = 0
        self.last_obs = o

        # in_space = {k: o[k] in s for k, s in self.observation_space.spaces.items()}

        return o, info

    def render(self):
        obs = self.last_obs

        with open(f"{self.domain}_{self.instance}_{self.iter}.dot", "w") as f:
            if self.render_mode == "human":
                f.write(
                    to_graphviz(
                        obs, self.idx_to_obj, self.idx_to_rel, self.object_to_type
                    )
                )
            else:
                f.write(
                    to_graphviz_dict(graph_to_dict(obs, self.types_instead_of_objects))
                )

    def step(self, action: tuple[int, int]):
        action_fluent = self.action_fluents[action[0]]
        object_id = self.idx_to_obj[action[1]]

        rddl_action = (
            f"{action_fluent}___{object_id}" if action_fluent != "noop" else "noop"
        )

        self.last_action = rddl_action

        invalid_action = rddl_action not in self.action_groundings

        if invalid_action:
            logger.warning(f"Invalid action: {rddl_action}")

        rddl_action_dict = (
            {} if invalid_action or action_fluent == "noop" else {rddl_action: 1}
        )

        obs, reward, terminated, truncated, info = self.env.step(rddl_action_dict)

        obs |= self.non_fluents_values
        if self.add_inverse_relations:
            obs |= {
                create_inverse_grounding(k): v
                for k, v in obs.items()
                if self.arities[predicate(k)] == 2
            }

        filtered_groundings = sorted(
            [
                g
                for g in self.groundings
                if g in obs
                and not skip_grounding_with_val(
                    g, self.variable_ranges, self.arities, obs[g]
                )
            ]
        )

        filtered_obs: dict[str, Any] = {k: obs[k] for k in filtered_groundings}

        action_mask = np.stack(
            [np.array([x for x in v.values()]) for v in self.action_mask.values()]
        )

        tabs, assocs = obs_to_db(filtered_obs, self.object_to_type)
        o = db_to_indices(
            tabs,
            assocs,
            self.object_to_index,
            self.type_to_idx,
            self.rel_to_idx,
        )

        info["action_mask"] = action_mask
        info["state"] = (tabs, assocs)

        self.iter += 1
        self.last_obs = o
        self.last_rddl_obs = obs

        return (
            o,
            reward,
            terminated,
            truncated,
            info,
        )


def main():
    instance = 1
    # domain = "Elevators_MDP_ippc2011"
    domain = "SysAdmin_MDP_ippc2011"
    # domain = "RecSim_ippc2023"
    # domain = "SkillTeaching_MDP_ippc2011"
    env = KGRDDLGraphWrapper(domain, instance, add_inverse_relations=False)
    obs, info = env.reset(seed=0)
    env.render()
    done = False
    time = 0
    sum_reward = 0
    while not done:
        time += 1
        action: tuple[int, int] = env.action_space.sample()
        action_fluent = action[0]
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


def register_env():
    env_id = "KGRDDLGraphWrapper-v0"
    gym.register(
        id=env_id,
        entry_point="wrappers.kg_wrapper:KGRDDLGraphWrapper",
    )
    return env_id


if __name__ == "__main__":
    main()
