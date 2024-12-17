import gymnasium as gym
import torch
from test_imitation_mdp import evaluate, save_eval_data

from gnn import Config, GraphAgent
from rddl import register_env


def test_agent():
    path = "conditional_bandit.rddl__ppo_gnn__0.pth"
    agent, config = GraphAgent.load_agent(path)
    domain = "rddl/conditional_bandit.rddl"
    instance = "rddl/conditional_bandit_i0.rddl"
    env: gym.Env = gym.make(
        register_env(),
        domain=domain,
        instance=instance,
        # add_inverse_relations=False,
        # types_instead_of_objects=False,
    )
    data = [evaluate(env, agent, 0) for i in range(3)]
    save_eval_data(data)


if __name__ == "__main__":
    test_agent()
