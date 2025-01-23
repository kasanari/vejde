from collections import deque
import json
from random import shuffle

import torch as th
from tqdm import tqdm
from regawa.gnn.data import (
    dict_to_obsdata,
    heterodict_to_obsdata,
    heterostatedata,
    heterostatedata_from_obslist,
    single_obs_to_heterostatedata,
)
from regawa.gnn.gnn_agent import Config, GraphAgent
from regawa.gnn.gnn_policies import ActionMode
from regawa.model.base_model import BaseModel
from regawa.rddl import register_env
from regawa.rddl.rddl_utils import rddl_ground_to_tuple
from regawa.rl.util import evaluate, update
from regawa.wrappers.utils import (
    create_graphs,
    create_obs_dict,
    from_dict_action,
    object_list,
)
import gymnasium as gym


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


def to_obsdata(rddl_obs: dict, model: BaseModel):
    g, _ = create_graphs(
        rddl_obs[0],
        model,
    )
    o = heterodict_to_obsdata(create_obs_dict(g, model))
    obj_to_idx = lambda x: g.boolean.factors.index(x)
    a = to_action(rddl_obs[1], obj_to_idx, model)
    return o, a


def get_agent(model: BaseModel):
    n_types = model.num_types
    n_relations = model.num_fluents
    n_actions = model.num_actions

    config = Config(
        n_types,
        n_relations,
        n_actions,
        layers=4,
        embedding_dim=4,
        activation=th.nn.Mish(),
        aggregation="sum",
        action_mode=ActionMode.ACTION_THEN_NODE,
    )

    agent = GraphAgent(
        config,
    )

    return agent


def get_rddl_data(data: list, model: BaseModel):
    data = [convert_episode(d) for d in data]
    rollout = [to_obsdata(s, model) for e in data for s in e]
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
    datafile = "/storage/GitHub/pyRDDLGym-prost/prost/out/sysadmin1/data_sysadmin_mdp_sysadmin_inst_mdp__2.json"
    domain = "SysAdmin_MDP_ippc2011"
    instance = 2
    seed = 1
    env_id = register_env()
    env: gym.Env = gym.make(env_id, domain=domain, instance=instance)
    model: BaseModel = env.unwrapped.env.model

    agent = get_agent(model)
    optimizer = th.optim.AdamW(agent.parameters(), lr=0.01, amsgrad=True)

    with open(datafile, "r") as f:
        data = json.load(f)

    expert_rewards = test_expert(env, data, seed, model)
    print(
        f"Expert Total reward: {sum(expert_rewards)}, Mean reward: {sum(expert_rewards) / len(expert_rewards)}"
    )

    expert_obs, expert_action = get_rddl_data(data, model)

    # batch_inds = list(range(0, len(expert_obs)))

    d = heterostatedata_from_obslist(expert_obs)
    avg_loss = 0.0
    num_gradient_steps = 1
    avg_grad_norm = 0.0
    pbar = tqdm()
    for _ in range(num_gradient_steps):
        # shuffle(batch_inds)
        # o = [expert_obs[i] for i in batch_inds]
        # a = [expert_action[i] for i in batch_inds]
        # for x, y in zip(o, a):
        loss, grad_norm = update(agent, optimizer, expert_action, d)
        pbar.update(1)
        avg_loss = avg_loss + (loss - avg_loss) / 2
        avg_grad_norm = avg_grad_norm + (grad_norm - avg_grad_norm) / 2
        pbar.set_description(f"Loss: {avg_loss:.3f}, Grad Norm: {avg_loss:.3f}")

    pbar.close()

    rewards, _, _ = evaluate(env, agent, seed, deterministic=False)

    print(f"Learner Total reward: {sum(rewards)}")


if __name__ == "__main__":
    test_saved_data()
