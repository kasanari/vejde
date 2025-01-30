import random


import gymnasium as gym
import numpy as np
import pytest
import torch as th


from regawa.gnn import Config, GraphAgent, ActionMode
from regawa.rl.util import evaluate, rollout, save_eval_data, update
from regawa.rddl import register_env
import regawa.model.utils as model_utils
import logging


def policy(state: dict[str, bool]) -> tuple[int, int]:
    if state[("light", "r_m")]:
        return (1, 4)

    if state[("light", "g_m")]:
        return (1, 2)

    return (0, 0)


@pytest.mark.parametrize(
    "action_mode, iterations",
    [
        (ActionMode.NODE_THEN_ACTION, 40),
        (ActionMode.ACTION_THEN_NODE, 30),
    ],
)
def test_imitation(action_mode: ActionMode, iterations: int):
    domain = "rddl/conditional_bandit.rddl"
    instance = "rddl/conditional_bandit_i0.rddl"

    th.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    l = logging.getLogger("regawa")
    l.setLevel(logging.INFO)
    logfile = logging.FileHandler("test_imitation_mdp.log", mode="w")
    l.addHandler(logfile)

    env_id = register_env()
    env: gym.Env = gym.make(env_id, domain=domain, instance=instance)

    # n_objects = env.observation_space.spaces["factor"].shape[0]
    # n_vars = env.observation_space["var_value"].shape[0]
    n_types = model_utils.n_types(env.observation_space)
    n_relations = model_utils.n_relations(env.observation_space)
    n_actions = model_utils.n_actions(env.action_space)

    config = Config(
        n_types,
        n_relations,
        n_actions,
        layers=4,
        embedding_dim=16,
        activation=th.nn.Mish(),
        aggregation="sum",
        action_mode=action_mode,
    )

    agent = GraphAgent(
        config,
    )

    # agent, config = agent.load_agent("conditional_bandit.pth")

    optimizer = th.optim.AdamW(agent.parameters(), lr=0.01, amsgrad=True)

    data = [evaluate(env, agent, 0) for i in range(10)]
    rewards, _, _ = zip(*data)
    print(np.mean(np.sum(rewards, axis=1)))

    losses = [iteration(i, env, agent, optimizer, 0) for i in range(iterations)]

    max_loss = 1e-6
    assert losses[-1] < max_loss, "Loss was too high: expected less than %s, got %s" % (
        max_loss,
        losses[-1],
    )

    pass

    data = [evaluate(env, agent, 0) for i in range(3)]
    rewards, _, _ = zip(*data)
    avg_reward = np.mean(np.sum(rewards, axis=1))
    print(avg_reward)

    assert avg_reward == 4.0, "Reward was too low: expected %s, got %s" % (
        4.0,
        avg_reward,
    )

    save_eval_data(data)

    # agent.save_agent("conditional_bandit.pth")


def iteration(i, env, agent, optimizer, seed: int):
    r, length = rollout(env, seed, policy, 4.0)

    # save_rollout(r, f"rollouts/rollout_{i}.json")
    # saved_r = load_rollout(f"rollouts/rollout_{i}.json")
    # compare_rollouts(r, saved_r)

    loss, grad_norm = update(agent, optimizer, r.actions, r.obs.batch)
    print(f"{i} Loss: {loss:.3f}, Grad Norm: {grad_norm:.3f}, Length: {length}")
    return loss


if __name__ == "__main__":
    test_imitation(ActionMode.NODE_THEN_ACTION, 70)
