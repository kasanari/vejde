from collections import deque
import json

import torch as th
from tqdm import tqdm
from regawa.gnn.agent_utils import GNNParams
from regawa.gnn.data import (
    heterodict_to_obsdata,
    heterostatedata_from_obslist,
)
from regawa.gnn.gnn_agent import AgentConfig, GraphAgent, heterostatedata_to_tensors
from regawa.gnn import ActionMode
from regawa.model.base_grounded_model import BaseGroundedModel
from regawa.model.base_model import BaseModel
from regawa.rddl import register_env
from regawa.rddl.rddl_utils import rddl_ground_to_tuple
from regawa.rl.util import calc_loss, evaluate, update
from regawa.wrappers.graph_utils import create_graphs
from regawa.wrappers.graph_utils import create_obs_dict
from regawa.wrappers.render_utils import create_render_graph, to_graphviz
from regawa.wrappers.utils import (
    from_dict_action,
    object_list,
)
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import random


def convert_to_tuples(d: dict):
    s = d["state"]
    a = d["actions"]

    s_t = {rddl_ground_to_tuple(k): v for k, v in s.items()}
    a = (
        {rddl_ground_to_tuple(k): v for k, v in a.items()}
        if a
        else {("None", "None"): True}
    )

    return s_t, a


def convert_episode(d: list[dict]):
    return list(map(convert_to_tuples, d))


def to_action(action: dict, obj_to_idx, model) -> tuple[int, int]:
    action_to_idx = lambda x: model.action_fluents.index(x)
    action = list(action.keys())[0]
    a = from_dict_action(action, action_to_idx, obj_to_idx)
    return a


render_index = 0


def to_obsdata(rddl_obs: dict, model: BaseModel, grounded_model: BaseGroundedModel):
    obs, action = rddl_obs
    obs |= {
        g: grounded_model.constant_value(g) for g in grounded_model.constant_groundings
    }

    g, _ = create_graphs(
        obs,
        model,
    )
    dot = to_graphviz(create_render_graph(g.boolean, g.numeric))
    global render_index
    with open(f"saved_render/graph_{render_index}.dot", "w") as f:
        f.write(dot)
    render_index += 1
    o = heterodict_to_obsdata(create_obs_dict(g, model))
    obj_to_idx = lambda x: g.boolean.factors.index(x)
    a = to_action(action, obj_to_idx, model)
    return o, a


def get_agent(model: BaseModel):
    n_types = model.num_types
    n_relations = model.num_fluents
    n_actions = model.num_actions

    params = GNNParams(
        layers=4,
        embedding_dim=32,
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


def get_rddl_data(data: list, model: BaseModel, grounded_model: BaseGroundedModel):
    data = [convert_episode(d) for d in data]
    rollout = [to_obsdata(s, model, grounded_model) for e in data for s in e]
    return zip(*rollout)


def test_expert(env, expert_data, seed, model):
    _, info = env.reset(seed=seed)
    done = False
    time = 0

    rewards = deque()

    step = 0
    while not done:
        time += 1

        rddl_state = info["rddl_state"]
        e_state = expert_data[0][step]["state"]
        # for k, v in e_state.items():
        #     assert rddl_state[rddl_ground_to_tuple(k)] == v

        action = expert_data[0][step]["actions"]
        action = {rddl_ground_to_tuple(list(action.keys())[0]): True}
        objs = object_list(rddl_state.keys(), model.fluent_param)
        objs = [o.name for o in objs]

        a = to_action(action, lambda x: objs.index(x), model)

        _, reward, terminated, truncated, info = env.step(a)  # type: ignore

        done = terminated or truncated
        rewards.append(reward)

    return rewards


def test_saved_data():
    # datafile = "/storage/GitHub/pyRDDLGym-prost/prost/out/sysadmin1/data_sysadmin_mdp_sysadmin_inst_mdp__2.json"
    # domain = "SysAdmin_MDP_ippc2011"
    # instance = 2
    datafile = "/storage/GitHub/pyRDDLGym-prost/prost/out/OUTPUTS/data_academic-advising_mdp_academic-advising_inst_mdp__01.json"
    domain = "AcademicAdvising_ippc2018"
    instance = 1

    seed = 1
    env_id = register_env()
    env: gym.Env = gym.make(env_id, domain=domain, instance=instance)
    model: BaseModel = env.unwrapped.model
    grounded_model: BaseGroundedModel = env.unwrapped.grounded_model

    np.random.seed(seed)
    th.manual_seed(seed)
    random.seed(seed)

    agent = get_agent(model)
    # agent, _ = GraphAgent.load_agent("saved_data.pth")
    optimizer = th.optim.AdamW(
        agent.parameters(), lr=0.01, amsgrad=True, weight_decay=0.01
    )

    with open(datafile, "r") as f:
        data = json.load(f)

    expert_rewards = test_expert(env, data, seed, model)
    print(
        f"Expert Total reward: {sum(expert_rewards)}, Mean reward: {sum(expert_rewards) / len(expert_rewards)}"
    )
    rewards, *_ = evaluate(env, agent, seed, deterministic=True)
    print(f"Learner Total reward: {sum(rewards)}")

    expert_obs, expert_action = get_rddl_data(data, model, grounded_model)

    # batch_inds = list(range(0, len(expert_obs)))

    d = heterostatedata_to_tensors(heterostatedata_from_obslist(expert_obs))
    avg_loss = 0.0
    num_gradient_steps = 400
    avg_grad_norm = 0.0
    pbar = tqdm()
    grad_norms = deque()
    losses = deque()
    for _ in range(num_gradient_steps):
        # shuffle(batch_inds)
        # o = [expert_obs[i] for i in batch_inds]
        # a = [expert_action[i] for i in batch_inds]
        # for x, y in zip(o, a):
        loss, grad_norm, _ = update(agent, optimizer, expert_action, d)
        pbar.update(1)
        avg_loss = avg_loss + (loss - avg_loss) / 2
        avg_grad_norm = avg_grad_norm + (grad_norm - avg_grad_norm) / 2
        grad_norms.append(grad_norm)
        losses.append(loss)
        pbar.set_description(f"Loss: {avg_loss:.3f}, Grad Norm: {avg_loss:.3f}")

    pbar.close()

    rewards, *_ = evaluate(env, agent, seed, deterministic=True)

    print(f"Learner Total reward: {sum(rewards)}")

    fig, axs = plt.subplots(2)
    axs[0].plot(list(losses))
    axs[1].plot(list(grad_norms))
    axs[0].set_title("Loss")
    axs[1].set_title("Grad Norm")
    fig.savefig("test_saved_data.png")

    agent.save_agent("saved_data.pth")

    loss_per_obs = []
    for d in expert_obs:
        s = heterostatedata_to_tensors(heterostatedata_from_obslist([d]))
        actions, logprob, entropy, value, p_n, p_a__n = agent.sample(s)
        l2_norms = [th.sum(th.square(w)) for w in agent.parameters()]
        loss = calc_loss(l2_norms, logprob)
        loss_per_obs.append(loss.item())

    sorted_loss = sorted(enumerate(loss_per_obs), key=lambda x: x[1], reverse=True)

    with open("sorted_loss.txt", "w") as f:
        for i, l in sorted_loss:
            f.write(f"{i}: {l}\n")

    pass


if __name__ == "__main__":
    test_saved_data()
