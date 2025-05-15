import json
import pathlib
from pathlib import Path
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
from regawa.gnn.data import heterostatedata_from_obslist
from regawa.gnn.gnn_agent import (
    AgentConfig,
    GraphAgent,
    RecurrentGraphAgent,
    heterostatedata_to_tensors,
)
from regawa.inference import fn_graph_to_obsdata, fn_groundobs_to_graph
from regawa.model.base_grounded_model import BaseGroundedModel
from regawa.model.base_model import BaseModel
from regawa.model.utils import max_arity
from regawa.rddl import register_env, register_pomdp_env
from regawa.rddl.rddl_utils import rddl_ground_to_tuple
from regawa.rl.util import calc_loss, evaluate, save_eval_data, update
from regawa.wrappers.graph_utils import fn_obsdict_to_graph
from regawa.wrappers.grounding_utils import fn_objects_with_type
from regawa.wrappers.remove_false_wrapper import remove_false
from regawa.wrappers.render_utils import create_render_graph
from regawa.wrappers.utils import from_dict_action, object_list

RecordingObs = dict[str, Any]
RecordingAction = dict[str, int]
RecordingEntry = dict[RecordingObs, RecordingAction]
Recording = list[RecordingEntry]

GroundAction = dict[GroundValue, bool]
GroundObs = dict[GroundValue, Any]

IndexedAction = tuple[int, ...]


@th.inference_mode()
def save_sorted_losses(
    model: BaseModel,
    agent: GraphAgent,
    expert_actions: list[GroundAction],
    indexed_expert_obs: list[GroundObs],
    expert_obs: list[GroundObs],
    device: str = "cpu",
):
    loss_per_obs = []
    objects_with_type = fn_objects_with_type(model.fluent_param)
    for i, (expert_a, d, o) in enumerate(
        zip(expert_actions, indexed_expert_obs, expert_obs)
    ):
        s = heterostatedata_to_tensors(heterostatedata_from_obslist([d]), device=device)
        g = to_graph(o, model)
        actions, logprob, _, _, p_a, p_n__a = agent.sample(s, deterministic=True)
        l2_norms = [th.sum(th.square(w)) for w in agent.parameters()]
        loss = calc_loss(l2_norms, logprob).item()

        objs = object_list(list(o.keys()), objects_with_type)
        objs = [o.name for o in objs]

        model_a = from_index_action(actions[0], lambda x: objs[x], model)

        factor_weights = p_n__a.T[actions[:, 0]].detach().squeeze().cpu().numpy()

        weight_by_factor = {
            k: f"{float(v):0.3f}"
            for k, v in zip(g.factor_labels, factor_weights)
            if v > 0.001
        }

        weight_by_action = {
            k: f"{float(v):0.3f}"
            for k, v in zip(
                model.action_fluents,
                p_a.detach().squeeze().cpu().numpy(),
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
                # obs=o,
            )
        )

    sorted_loss = sorted(loss_per_obs, key=lambda x: x["loss"], reverse=True)

    return sorted_loss


def ground_to_tuple(s: str) -> GroundValue:
    return rddl_ground_to_tuple(s)


def convert_state_to_tuples(
    d: RecordingObs, converter_func: Callable[[str], GroundValue]
) -> dict[GroundValue, Any]:
    return {converter_func(k): v for k, v in d.items()}


def convert_actions_to_tuples(
    d: RecordingAction, converter_func: Callable[[str], GroundValue]
) -> dict[GroundValue, bool]:
    return convert_state_to_tuples(d, converter_func) if d else {("None", "None"): True}


def ensure_tuple(x: tuple[str, ...]) -> tuple[str, ...]:
    return x + ("None",) if len(x) == 1 else x


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
    create_graph = fn_obsdict_to_graph(model)
    g, _ = create_graph(obs)
    return create_render_graph(g.boolean, g.numeric)


def to_obsdata(
    obs: GroundObs,
    action: GroundAction,
    model: BaseModel,
):
    create_graphs = fn_obsdict_to_graph(model)
    g_to_obsdata = fn_graph_to_obsdata(model)
    g, _ = create_graphs(
        obs,
    )
    o = g_to_obsdata(g)
    a = to_indexed_action(action, lambda x: g.boolean.factors.index(x), model)

    # Rendering
    # dot = to_graphviz(create_render_graph(g.boolean, g.numeric))
    # global render_index
    # render_path = pathlib.Path("saved_render")
    # render_path.mkdir(exist_ok=True)
    # with open(render_path / f"graph_{render_index}.dot", "w") as f:
    #     f.write(dot)
    # render_index += 1

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


def get_agent(model: BaseModel, device: str = "cpu"):
    n_types = model.num_types
    n_relations = model.num_fluents
    n_actions = model.num_actions
    arity = max_arity(model)

    params = GNNParams(
        layers=4,
        embedding_dim=16,
        activation=th.nn.Tanh(),
        aggregation="max",
        action_mode=ActionMode.NODE_THEN_ACTION,
    )

    config = AgentConfig(
        n_types,
        n_relations,
        n_actions,
        hyper_params=params,
        arity=arity,
        remove_false_fluents=True,
    )

    agent = GraphAgent(config, None)

    agent = agent.to(device)

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


def test_saved_data(domain: str, data_path: str):
    datafile = Path(f"{data_path}/{domain}/combined_data.json").expanduser()
    instance = 1
    use_rnn = False
    seed = 1
    device = "cuda:0" if th.cuda.is_available() else "cpu"
    num_gradient_steps = 500
    env_id = register_pomdp_env() if use_rnn else register_env()

    output_dir = pathlib.Path("imitation_output")
    output_dir.mkdir(exist_ok=True)
    domain_dir = output_dir / domain
    domain_dir.mkdir(exist_ok=True)

    env: gym.Env = gym.make(env_id, domain=domain, instance=instance, remove_false=True)
    model: BaseModel = env.unwrapped.model

    np.random.seed(seed)
    th.manual_seed(seed)
    random.seed(seed)

    agent = get_rnn_agent(model) if use_rnn else get_agent(model, device)

    # agent, _ = GraphAgent.load_agent(
    #     "imitation_output/Elevators_MDP_ippc2014/model.pth"
    # )

    optimizer = th.optim.AdamW(
        agent.parameters(), lr=0.001, amsgrad=True, weight_decay=0.0
    )

    with open(datafile, "r") as f:
        expert_data = json.load(f)

    data = [
        evaluate(env, agent, i, deterministic=True, device=device) for i in range(10)
    ]
    rewards, *_ = zip(*data)

    print(f"Learner average return: {np.mean([sum(r) for r in rewards])}")

    expert_actions = [x["actions"] for x in expert_data]
    expert_obs = [x["state"] for x in expert_data]

    to_tuple = lambda x: tuple(x.split("__"))
    wrapper_func = lambda x: remove_false(
        convert_state_to_tuples(
            x,
            to_tuple,
        )
    )
    expert_actions = [convert_actions_to_tuples(e, to_tuple) for e in expert_actions]
    expert_actions = [
        {ensure_tuple(k): v} for e in expert_actions for k, v in e.items()
    ]
    expert_obs = [wrapper_func(e) for e in expert_obs]

    indexed_expert_obs, indexed_expert_action = zip(
        *[to_obsdata(o, a, model) for o, a in zip(expert_obs, expert_actions)]
    )

    d = heterostatedata_to_tensors(
        heterostatedata_from_obslist(indexed_expert_obs), device=device
    )
    indexed_expert_action = th.as_tensor(
        indexed_expert_action, dtype=th.int64, device=device
    )
    avg_loss = 0.0
    avg_grad_norm = 0.0
    pbar = tqdm()
    grad_norms = deque()
    losses = deque()
    for _ in range(num_gradient_steps):
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

    data = [
        evaluate(env, agent, i, deterministic=True, device=device) for i in range(10)
    ]
    rewards, *_ = zip(*data)
    save_eval_data(data, domain_dir / "eval_data.json")

    print(f"Learner average return: {np.mean([sum(r) for r in rewards])}")

    fig, axs = plt.subplots(2)
    axs[0].plot(list(losses))
    axs[1].plot(list(grad_norms))
    axs[0].set_title("Loss")
    axs[1].set_title("Grad Norm")
    fig.savefig(domain_dir / "loss_grad.png")

    agent.save_agent(str(domain_dir / "model.pth"))

    sorted_losses = save_sorted_losses(
        model, agent, expert_actions, indexed_expert_obs, expert_obs, device=device
    )
    with open(domain_dir / "sorted_loss.json", "w") as f:
        json.dump(sorted_losses, f, indent=2)

    pass


if __name__ == "__main__":
    # domains = "Navigation_MDP_ippc2011 TriangleTireworld_MDP_ippc2014 Elevators_MDP_ippc2014 SysAdmin_MDP_ippc2011 Traffic_MDP_ippc2014 SkillTeaching_MDP_ippc2014 AcademicAdvising_MDP_ippc2014 CrossingTraffic_MDP_ippc2014 Tamarisk_MDP_ippc2014"
    # domains = domains.split()
    domains = ["Elevators_MDP_ippc2014"]

    import sys

    # data_path = sys.argv[1]
    # domains = ["SysAdmin_MDP_ippc2011"]
    # domains = ["Tamarisk_MDP_ippc2014"]
    for domain in domains:
        print(f"Testing {domain}")
        data_path = Path("/storage/GitHub/pyRDDLGym-rl/prost/").expanduser()
        test_saved_data(domain, data_path)
