import numpy as np

from wrappers.kg_wrapper import KGRDDLGraphWrapper


def test_kg():
    instance = 1
    domain = "Elevators_MDP_ippc2011"
    # domain = "SysAdmin_MDP_ippc2011"
    # domain = "RecSim_ippc2023"
    # domain = "SkillTeaching_MDP_ippc2011"
    env = KGRDDLGraphWrapper(domain, instance, add_inverse_relations=False)
    obs, info = env.reset(seed=0)
    env.render()
    done = False
    time = 0
    sum_reward = 0
    while not done:
        time += 1
        action: tuple[int, int] = env.action_space.sample()
        action_fluent = action[0]
        action_mask = info["action_mask"]
        valid_objects = np.flatnonzero(action_mask[action_fluent])
        object_idx = np.random.choice(valid_objects)
        new_action = (object_idx, action_fluent)
        obs, reward, terminated, truncated, info = env.step(new_action)
        env.render()
        # exit()
        done = terminated or truncated

        sum_reward += reward
        # print(obs)
        print(new_action)
        print(reward)
    print(sum_reward)


if __name__ == "__main__":
    test_kg()
