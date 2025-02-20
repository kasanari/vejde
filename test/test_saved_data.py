import json
import pathlib
import random
from collections import deque
from collections.abc import Callable
from typing import Any, SupportsFloat

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from tqdm import tqdm

from regawa import GNNParams, GroundValue
from regawa.gnn import ActionMode
from regawa.gnn.data import heterodict_to_obsdata, heterostatedata_from_obslist
from regawa.gnn.gnn_agent import (
    AgentConfig,
    GraphAgent,
    RecurrentGraphAgent,
    heterostatedata_to_tensors,
)
from regawa.model.base_grounded_model import BaseGroundedModel
from regawa.model.base_model import BaseModel
from regawa.rddl import register_env
from regawa.rddl.rddl_utils import rddl_ground_to_tuple
from regawa.rl.util import calc_loss, evaluate, update
from regawa.wrappers.graph_utils import create_graphs, create_obs_dict
from regawa.wrappers.render_utils import create_render_graph, to_graphviz
from regawa.wrappers.utils import from_dict_action, object_list

RecordingObs = dict[str, Any]
RecordingAction = dict[str, int]
RecordingEntry = dict[RecordingObs, RecordingAction]
Recording = list[RecordingEntry]

GroundAction = dict[GroundValue, bool]
GroundObs = dict[GroundValue, Any]

IndexedAction = tuple[int, ...]


def save_sorted_losses(
    model: BaseModel,
    agent: GraphAgent,
    expert_actions: list[GroundAction],
    indexed_expert_obs: list[GroundObs],
    expert_obs: list[GroundObs],
):
    loss_per_obs = []
    for i, (expert_a, d, o) in enumerate(
        zip(expert_actions[0], indexed_expert_obs, expert_obs[0])
    ):
        s = heterostatedata_to_tensors(heterostatedata_from_obslist([d]))
        g = to_graph(o, model)
        actions, logprob, _, _, p_a, p_n__a = agent.sample(s, deterministic=True)
        l2_norms = [th.sum(th.square(w)) for w in agent.parameters()]
        loss = calc_loss(l2_norms, logprob).item()

        objs = object_list(o.keys(), model.fluent_param)
        objs = [o.name for o in objs]

        model_a = from_index_action(actions[0], lambda x: objs[x], model)

        factor_weights = p_n__a.T[actions[:, 0]].detach().squeeze().numpy()

        weight_by_factor = {
            k: f"{float(v):0.3f}"
            for k, v in zip(g.factor_labels, factor_weights)
            if v > 0.001
        }

        weight_by_action = {
            k: f"{float(v):0.3f}"
            for k, v in zip(
                model.action_fluents,
                p_a.detach().squeeze().numpy(),
            )
            if v > 0.001
        }

        loss_per_obs.append(
            dict(
                model_action=model_a,
                expert_action=list(expert_a.keys())[0],
                loss=loss,
                action_probs=weight_by_action,
                object_probs=weight_by_factor,
                step=i,
            )
        )

    sorted_loss = sorted(loss_per_obs, key=lambda x: x["loss"], reverse=True)

    with open("sorted_loss.json", "w") as f:
        json.dump(sorted_loss, f, indent=2)


def ground_to_tuple(s: str) -> GroundValue:
    return rddl_ground_to_tuple(s)


def convert_state_to_tuples(d: RecordingObs) -> dict[GroundValue, Any]:
    return {ground_to_tuple(k): v for k, v in d.items() if v}


def convert_actions_to_tuples(d: RecordingAction) -> dict[GroundValue, bool]:
    return convert_state_to_tuples(d) if d else {("None", "None"): True}


def from_index_action(
    action: IndexedAction, idx_to_obj: Callable[[int], str], model: BaseModel
) -> tuple[str, str]:
    return (model.action_fluents[action[0]], idx_to_obj(action[1]))


def to_indexed_action(
    action: GroundAction, obj_to_idx: Callable[[str], int], model: BaseModel
) -> IndexedAction:
    action = list(action.keys())[0] if action else ("None", "None")
    a = from_dict_action(action, lambda x: model.action_fluents.index(x), obj_to_idx)
    return a


render_index = 0


def get_actions(data: Recording):
    return [[x["actions"] for x in d] for d in data]


def get_obs(data: Recording):
    return [[x["state"] for x in d] for d in data]


def to_graph(obs: GroundObs, model: BaseModel):
    g, _ = create_graphs(obs, model)
    return create_render_graph(g.boolean, g.numeric)


def to_obsdata(
    obs: GroundObs,
    action: GroundAction,
    model: BaseModel,
    grounded_model: BaseGroundedModel,
):
    obs |= {
        g: grounded_model.constant_value(g) for g in grounded_model.constant_groundings
    }

    g, _ = create_graphs(
        obs,
        model,
    )
    dot = to_graphviz(create_render_graph(g.boolean, g.numeric))
    global render_index
    render_path = pathlib.Path("saved_render")
    render_path.mkdir(exist_ok=True)
    with open(render_path / f"graph_{render_index}.dot", "w") as f:
        f.write(dot)
    render_index += 1
    o = heterodict_to_obsdata(create_obs_dict(g, model))
    a = to_indexed_action(action, lambda x: g.boolean.factors.index(x), model)
    return o, a


def get_rnn_agent(model: BaseModel):
    n_types = model.num_types
    n_relations = model.num_fluents
    n_actions = model.num_actions

    params = GNNParams(
        layers=4,
        embedding_dim=16,
        activation=th.nn.Mish(),
        aggregation="sum",
        action_mode=ActionMode.NODE_THEN_ACTION,
    )

    config = AgentConfig(
        n_types,
        n_relations,
        n_actions,
        params,
    )

    agent = RecurrentGraphAgent(
        config,
    )

    return agent


def get_agent(model: BaseModel):
    n_types = model.num_types
    n_relations = model.num_fluents
    n_actions = model.num_actions

    params = GNNParams(
        layers=4,
        embedding_dim=16,
        activation=th.nn.Mish(),
        aggregation="sum",
        action_mode=ActionMode.ACTION_THEN_NODE,
    )

    config = AgentConfig(
        n_types,
        n_relations,
        n_actions,
        params,
    )

    agent = GraphAgent(
        config,
    )

    return agent


def get_rddl_data(data: Recording, model: BaseModel, grounded_model: BaseGroundedModel):
    data = [convert_episode(d) for d in data]
    rollout = [to_obsdata(s, model, grounded_model) for e in data for s in e]
    return zip(*rollout)


def test_expert(
    env: gym.Env,
    expert_data: Recording,
    seed: int,
    model: BaseModel,
    strict: bool = True,
):
    _, info = env.reset(seed=seed)
    done = False

    rewards: deque[SupportsFloat] = deque()

    step = 0
    while not done:
        rddl_state = info["rddl_state"]
        e_state = expert_data[0][step]["state"]
        for k, v in e_state.items():
            tuple_k = ground_to_tuple(k)
            assert tuple_k in rddl_state, (
                "Grounded value %s from recording not in state returned by the simulator at step %d"
                % (k, step)
            )
            obs_equal = (
                (
                    rddl_state[tuple_k][-1] == v
                    if isinstance(rddl_state[tuple_k], list)
                    else rddl_state[tuple_k] == v
                )
                if strict
                else True
            )
            assert obs_equal, (
                "Expected value %s for grounded value %s but %s was returned by the simulator"
                % (v, k, rddl_state[tuple_k])
            )

        action = expert_data[0][step]["actions"]
        action = {ground_to_tuple(list(action.keys())[0]): True} if action else {}
        objs = object_list(rddl_state.keys(), model.fluent_param)
        objs = [o.name for o in objs]

        a = to_indexed_action(action, lambda x: objs.index(x), model)

        _, reward, terminated, truncated, info = env.step(a)  # type: ignore

        done = terminated or truncated
        rewards.append(reward)
        step += 1

    return rewards


def test_saved_data():
    # datafile = "/storage/GitHub/pyRDDLGym-prost/prost/out/sysadmin1/data_sysadmin_mdp_sysadmin_inst_mdp__2.json"
    # domain = "SysAdmin_MDP_ippc2011"
    # instance = 2
    datafile = "/storage/GitHub/pyRDDLGym-prost/prost/out/OUTPUTS/data_academic-advising_mdp_academic-advising_inst_mdp__01.json"
    domain = "AcademicAdvising_ippc2018"
    instance = 1
    use_rnn = False
    seed = 1
    env_id = register_stacking_env() if use_rnn else register_env()
    env: gym.Env = gym.make(env_id, domain=domain, instance=instance)
    model: BaseModel = env.unwrapped.model
    grounded_model: BaseGroundedModel = env.unwrapped.grounded_model
    np.random.seed(seed)
    th.manual_seed(seed)
    random.seed(seed)

    agent = get_rnn_agent(model) if use_rnn else get_agent(model)
    optimizer = th.optim.AdamW(
        agent.parameters(), lr=0.01, amsgrad=True, weight_decay=0.01
    )

    with open(datafile, "r") as f:
        data = json.load(f)

    expert_rewards = test_expert(env, data, seed, model, strict=False)
    print(
        f"Expert Total reward: {sum(expert_rewards)}, Mean reward: {sum(expert_rewards) / len(expert_rewards)}"
    )
    rewards, *_ = evaluate(env, agent, seed, deterministic=True)
    print(f"Learner Total reward: {sum(rewards)}")

    expert_actions = get_actions(data)
    expert_obs = get_obs(data)
    expert_actions = [list(map(convert_actions_to_tuples, e)) for e in expert_actions]
    expert_obs = [list(map(convert_state_to_tuples, e)) for e in expert_obs]

    indexed_expert_obs, indexed_expert_action = zip(
        *[
            to_obsdata(o, a, model, grounded_model)
            for e_o, e_a in zip(expert_obs, expert_actions)
            for o, a in zip(e_o, e_a)
        ]
    )

    # batch_inds = list(range(0, len(expert_obs)))

    d = heterostatedata_to_tensors(heterostatedata_from_obslist(indexed_expert_obs))
    avg_loss = 0.0
    num_gradient_steps = 500
    avg_grad_norm = 0.0
    pbar = tqdm()
    grad_norms = deque()
    losses = deque()
    for _ in range(num_gradient_steps):
        # shuffle(batch_inds)
        # o = [expert_obs[i] for i in batch_inds]
        # a = [expert_action[i] for i in batch_inds]
        # for x, y in zip(o, a):
        loss, grad_norm, _ = update(
            agent, optimizer, indexed_expert_action, d, max_grad_norm=0.5
        )
        pbar.update(1)
        avg_loss = avg_loss + (loss - avg_loss) / 2
        avg_grad_norm = avg_grad_norm + (grad_norm - avg_grad_norm) / 2
        grad_norms.append(grad_norm)
        losses.append(loss)
        pbar.set_description(f"Loss: {avg_loss:.3f}, Grad Norm: {avg_loss:.3f}")

    pbar.close()

    rewards, *_ = evaluate(env, agent, seed, deterministic=False)

    print(f"Learner Total reward: {sum(rewards)}")

    fig, axs = plt.subplots(2)
    axs[0].plot(list(losses))
    axs[1].plot(list(grad_norms))
    axs[0].set_title("Loss")
    axs[1].set_title("Grad Norm")
    fig.savefig("test_saved_data.png")

    agent.save_agent("saved_data.pth")

    save_sorted_losses(model, agent, expert_actions, indexed_expert_obs, expert_obs)

    pass


if __name__ == "__main__":
    test_saved_data()
