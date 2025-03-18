import logging
import random
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch as th
import torch.optim as optim

import regawa.wrappers.gym_utils as model_utils
from regawa.gnn import ActionMode, AgentConfig, GraphAgent
from regawa.gnn.agent_utils import GNNParams
from regawa.gnn.gnn_agent import heterostatedata_to_tensors
from regawa.rddl import register_env
from regawa.rl.util import evaluate, rollout, save_eval_data, update, update_vf_agent


def policy(state: dict[str, bool]) -> tuple[int, int]:
    if state.get(("light", "r_m"), False):
        return (1, 4)

    if state.get(("light", "g_m"), False):
        return (1, 2)

    return (0, 0)


def plot_per_grad_norms(per_param_grad, action_mode):
    keys = per_param_grad[0].keys()
    per_param_grad = {k: [d[k] for d in per_param_grad] for k in keys}

    fig, axs = plt.subplots(len(per_param_grad), figsize=(10, 30))
    for i, (k, v) in enumerate(per_param_grad.items()):
        axs[i].plot(v, "-o")
        axs[i].set_title(k)
        # axs[i].set_ylim(0, 5)
    fig.tight_layout()
    fig.savefig(f"test_imitation_mdp_{action_mode}_grads.pdf")


def plot_loses_grads(losses, norms, action_mode):
    fig, axs = plt.subplots(2)
    axs[0].plot(losses)
    axs[1].plot(norms)
    axs[0].set_title("Loss")
    axs[1].set_title("Grad Norm")
    fig.savefig(f"test_imitation_mdp_{action_mode}.png")


@pytest.mark.parametrize(
    "action_mode, iterations, embedding_dim",
    [
        (ActionMode.NODE_THEN_ACTION, 14, 16),
        (ActionMode.ACTION_THEN_NODE, 13, 16),
    ],
)
def test_imitation(
    action_mode: ActionMode,
    iterations: int,
    embedding_dim: int,
    remove_false: bool = False,
):
    domain = "rddl/conditional_bandit.rddl"
    instance = "rddl/conditional_bandit_i0.rddl"

    th.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    l = logging.getLogger("regawa")
    l.setLevel(logging.INFO)
    logfile = logging.FileHandler("test_imitation_mdp.log", mode="w")
    l.addHandler(logfile)

    render_logger = logging.getLogger("message_pass_render")
    render_logfile = logging.FileHandler("test_imitation_mdp_render.log", mode="w")
    render_logger.addHandler(render_logfile)

    env_id = register_env()
    env: gym.Env = gym.make(
        env_id, domain=domain, instance=instance, remove_false=remove_false
    )

    # n_objects = env.observation_space.spaces["factor"].shape[0]
    # n_vars = env.observation_space["var_value"].shape[0]
    n_types = model_utils.n_types(env.observation_space)
    n_relations = model_utils.n_relations(env.observation_space)
    n_actions = model_utils.n_actions(env.action_space)

    params = GNNParams(
        layers=4,
        embedding_dim=embedding_dim,
        activation=th.nn.Mish(),
        aggregation="max",
        action_mode=action_mode,
    )

    config = AgentConfig(
        n_types,
        n_relations,
        n_actions,
        remove_false_fluents=remove_false,
        arity=model_utils.max_arity(env.observation_space),
        hyper_params=params,
    )

    rng = th.Generator()

    agent = GraphAgent(
        config,
        rng,
    )
    vf_agent = GraphAgent(
        config,
        rng,
    )

    # agent, config = agent.load_agent("conditional_bandit.pth")

    optimizer = optim.AdamW(agent.parameters(), lr=0.01, amsgrad=True, weight_decay=0.1)
    vf_optimizer = optim.AdamW(
        vf_agent.parameters(), lr=0.01, amsgrad=True, weight_decay=0.01
    )

    data = [evaluate(env, agent, 0) for i in range(10)]
    rewards, *_ = zip(*data)
    before_training_rewards = np.mean(np.sum(rewards, axis=1))
    print(before_training_rewards)

    data = [
        iteration(i, env, agent, optimizer, vf_agent, vf_optimizer, 0)
        for i in range(iterations)
    ]

    losses, norms, per_param_grad = zip(*data)
    # reshape

    plot_loses_grads(losses, norms, action_mode)
    plot_per_grad_norms(per_param_grad, action_mode)

    max_loss = 1e-6
    assert losses[-1] < max_loss, "Loss was too high: expected less than %s, got %s" % (
        max_loss,
        losses[-1],
    )

    pass

    print(f"Trainable Parameters: {agent.num_trainable_params()}")

    render_logger.setLevel(logging.DEBUG)
    data = [evaluate(env, agent, 0) for i in range(3)]
    rewards, *_ = zip(*data)
    avg_reward = np.mean(np.sum(rewards, axis=1))
    print(avg_reward)

    assert avg_reward == 4.0, "Reward was too low: expected %s, got %s" % (
        4.0,
        avg_reward,
    )

    save_eval_data(data)

    # agent.save_agent("conditional_bandit.pth")


def iteration(i, env, agent, optimizer, vf_agent, vf_optimizer, seed: int):
    r, length = rollout(env, seed, policy, 4.0)

    # save_rollout(r, f"rollouts/rollout_{i}.json")
    # saved_r = load_rollout(f"rollouts/rollout_{i}.json")
    # compare_rollouts(r, saved_r)

    b = heterostatedata_to_tensors(r.obs.batch)
    loss, grad_norm, per_param_grad = update(agent, optimizer, r.actions, b)
    vf_loss, vf_grad_norm = update_vf_agent(vf_agent, vf_optimizer, b, r.rewards)
    print(
        f"{i} Loss: {loss:.3f}, Grad Norm: {grad_norm:.3f}, Length: {length}, VF Loss: {vf_loss:.3f}, VF Grad Norm: {vf_grad_norm:.3f}"
    )
    return loss, grad_norm, per_param_grad


if __name__ == "__main__":
    t = time.time()
    test_imitation(ActionMode.ACTION_THEN_NODE, 30, 16)
    print("Time: ", time.time() - t)
