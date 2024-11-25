from rddlrepository.core.manager import RDDLRepoManager
import pyRDDLGym
import yaml

manager = RDDLRepoManager(rebuild=True)

domain_arities = {}

action_arities = {}

for p in manager.list_problems():
    instance_arities = {}
    instance_action_arities = {}
    for i in manager.get_problem(p).list_instances():
        try:
            env = pyRDDLGym.make(p, i)
        except Exception as e:
            print(f"Error in {p} {i}: {e}")
            continue

        arities: dict[str, int] = {
            key: len(value) for key, value in env.model.variable_params.items()
        }

        action_fluents = env.model.action_fluents.keys()
        instance_action_arities[i] = max([arities[af] for af in action_fluents])
        instance_arities[i] = max(arities.values())

    if instance_arities:
        domain_arities[p] = max(instance_arities.values())
    if instance_action_arities:
        action_arities[p] = max(instance_action_arities.values())

filtered_arities = {k: v for k, v in domain_arities.items() if v <= 2 and v > 0}

filtered_action_arities = {
    k: v for k, v in action_arities.items() if k in filtered_arities
}

with open("arities.yml", "w") as f:
    yaml.dump(filtered_arities, f)

with open("action_arities.yml", "w") as f:
    yaml.dump(filtered_action_arities, f)

descriptions = {d: manager.get_problem(d).desc for d in domains}

with open("descriptions.yml", "w") as f:
    yaml.dump(descriptions, f)
