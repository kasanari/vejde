import random
from parent_wrapper import RDDLGraphWrapper
from utils import generate_bipartite_obs, to_graphviz


class GroundedRDDLGraphWrapper(RDDLGraphWrapper):
    def reset(self, seed: int | None = None):
        obs, info = self.env.reset(seed)

        obs |= self.action_values
        obs |= self.non_fluents_values

        (
            predicate_classes,
            predicate_values,
            object_nodes,
            edge_indices,
            edge_attributes,
            numeric,
        ) = generate_bipartite_obs(
            obs,
            self.groundings,
            self.symb_to_idx,
            self.variable_ranges,
        )

        obs = {
            "predicate_class": predicate_classes,
            "predicate_value": predicate_values,
            "object": object_nodes,
            "numeric": numeric,
            "edge_index": edge_indices,
            "edge_attr": edge_attributes,
        }

        self.iter = 0
        self.last_obs = obs

        info["symbol_list"] = self.idx_to_symb

        return obs, info

    def step(self, action: dict[str, int]):
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs |= self.action_values
        obs |= self.non_fluents_values

        (
            predicate_classes,
            predicate_values,
            object_nodes,
            edge_indices,
            edge_attributes,
            numeric,
        ) = generate_bipartite_obs(
            obs,
            self.groundings,
            self.symb_to_idx,
            self.variable_ranges,
        )

        obs = {
            "predicate_class": predicate_classes,
            "predicate_value": predicate_values,
            "object": object_nodes,
            "numeric": numeric,
            "edge_index": edge_indices,
            "edge_attr": edge_attributes,
        }

        self.iter += 1
        self.last_obs = obs

        info["symbol_list"] = self.idx_to_symb

        return obs, reward, terminated, truncated, info


def main():
    instance = 1
    domain = "Elevators_MDP_ippc2011"
    # domain = "SysAdmin_MDP_ippc2011"
    # domain = "RecSim_ippc2023"
    # domain = "SkillTeaching_MDP_ippc2011"
    env = GroundedRDDLGraphWrapper(domain, instance)
    obs, info = env.reset()
    env.render()
    done = False
    time = 0
    sum_reward = 0
    while not done:
        time += 1
        action = env.env.action_space.sample()

        action = random.choice(list(action.items()))
        action = {action[0]: action[1]}
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        exit()
        done = terminated or truncated

        sum_reward += reward
        # print(obs)
        print(action)
        print(reward)
    print(sum_reward)


if __name__ == "__main__":
    main()
