import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict, MultiDiscrete

from regawa.rddl import register_env
from regawa.rl.graph_buffer import ReplayBuffer


def make_env(
    env_id: str,
    domain: str,
    instance: str | int,
):
    def f() -> gym.Env[Dict, MultiDiscrete]:
        env: gym.Env[Dict, MultiDiscrete] = gym.make(  # type: ignore
            env_id,
            domain=domain,
            instance=instance,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return f


def test_buffer():
    domain: str = "rddl/conditional_bandit.rddl"
    instance: str | int = "rddl/conditional_bandit_i0.rddl"
    num_envs: int = 2
    env_id = register_env()
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, domain, instance) for _ in range(num_envs)],
    )

    buffer = ReplayBuffer(
        1024, envs.single_observation_space, envs.single_action_space, "cpu", num_envs
    )

    obs, info = envs.reset()

    for _ in range(1024):
        action = envs.action_space.sample()

        next_obs, reward, term, trunc, info = envs.step(action)

        done = np.logical_or(term, trunc)

        buffer.add(obs, next_obs, action, reward, done, info)

    batch = buffer.sample(32)

    print(batch)


if __name__ == "__main__":
    test_buffer()
