from gnn import GraphAgent, Config, ActionMode, StateData
import torch
from torch.nn import Module
from test_imitation_mdp import evaluate, save_eval_data
import gymnasium as gym
from wrappers.wrapper import register_env


def load_agent(path: str) -> tuple[GraphAgent, Config]:
    data = torch.load(path, weights_only=False)
    config = Config(**data["config"])
    agent = GraphAgent(config)
    agent.load_state_dict(
        {k.replace("agent.", ""): v for k, v in data["state_dict"].items()}
    )

    return agent, config


def test_agent():
    path = "conditional_bandit.rddl__ppo_gnn__1.pth"
    agent, config = load_agent(path)
    domain = "rddl/conditional_bandit.rddl"
    instance = "rddl/conditional_bandit_i0.rddl"
    env: gym.Env = gym.make(
        register_env(),
        domain=domain,
        instance=instance,
        # add_inverse_relations=False,
        # types_instead_of_objects=False,
        render_mode="idx",
    )
    data = [evaluate(env, agent, i) for i in range(3)]
    save_eval_data(data)


if __name__ == "__main__":
    test_agent()
