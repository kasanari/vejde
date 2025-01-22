import json
from random import shuffle

import torch as th
from tqdm import tqdm
from regawa.gnn.data import (
    dict_to_obsdata,
    heterodict_to_obsdata,
    heterostatedata,
    heterostatedata_from_obslist,
)
from regawa.gnn.gnn_agent import Config, GraphAgent
from regawa.gnn.gnn_policies import ActionMode
from regawa.model.base_model import BaseModel
from regawa.rddl import register_env
from regawa.rddl.rddl_utils import rddl_ground_to_tuple
from regawa.rl.util import evaluate, update
from regawa.wrappers.utils import create_obs, from_dict_action
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


def to_obsdata(d: dict, model: BaseModel):
    obs, g, _ = create_obs(
        d[0],
        model,
    )

    obj_to_idx = lambda x: g.boolean.factors.index(x)
    action_to_idx = lambda x: model.action_fluents.index(x)

    action = list(d[1].keys())[0]

    a = from_dict_action(action, action_to_idx, obj_to_idx)

    o = heterodict_to_obsdata(obs)

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


def get_rddl_data(datafile: str, model: BaseModel):
    with open(datafile, "r") as f:
        data = json.load(f)

    data = [convert_episode(d) for d in data]
    rollout = [to_obsdata(s, model) for e in data for s in e]
    return zip(*rollout)


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

    expert_obs, expert_action = get_rddl_data(datafile, model)

    # batch_inds = list(range(0, len(expert_obs)))

    d = heterostatedata_from_obslist(expert_obs)
    avg_loss = 0.0
    num_gradient_steps = 500
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

    print(f"Total reward: {sum(rewards)}")


if __name__ == "__main__":
    test_saved_data()
