import gymnasium as gym
import numpy as np

from regawa import GraphAgentInterface
from regawa.rl.util import evaluate


def evaluate_agent(
    agent: GraphAgentInterface,
    env_id: str,
    device: str,
):
    eval_env = gym.make(  # type: ignore
        env_id,
    )

    seeds = range(10)

    data = [evaluate(eval_env, agent, seed, rng=None, device=device) for seed in seeds]
    rewards, *_ = zip(*data, strict=False)
    avg_mean_reward = np.mean([np.mean(r) for r in rewards])
    returns = [np.sum(r).item() for r in rewards]

    stats = {
        "return_mean": np.mean(returns).item(),
        "return_median": np.median(returns).item(),
        "return_min": np.min(returns).item(),
        "return_max": np.max(returns).item(),
        "return_std": np.std(returns).item(),
        "mean_reward": avg_mean_reward.item(),
    }

    stats["returns"] = returns
    return stats, data
