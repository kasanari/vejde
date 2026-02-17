import sys
from pathlib import Path

import tyro

from regawa.data import create_render_graph, fn_groundobs_to_heterograph, to_graphviz
from regawa.model.base_grounded_model import GroundObs
from regawa.model.base_model import BaseModel
from regawa.model.io import step_from_json
from regawa.model.model_func import model_from_json

# test_model_path = "tyrLang_regawa_model.json"
# test_obs_path = "single_obs.json"


def render_obs(model: BaseModel, obs: GroundObs):
    obs_to_graph = fn_groundobs_to_heterograph(model, False)
    hetero_graph = obs_to_graph(obs)
    render_graph = create_render_graph(hetero_graph.boolean, hetero_graph.numeric)
    return to_graphviz(render_graph, pprint=False)


def filter_obs(obs: GroundObs):
    return {
        k: v for k, v in obs.items() if "Identity" not in k[0] and "Vuln" not in k[0]
    }


def main(model_path: Path, obs_path: Path):
    with open(model_path) as f:
        model_json = f.read()
    model = model_from_json(model_json)

    # check if json or jsonl
    multi_line = obs_path.suffix == ".jsonl"
    # print to stderr
    print(
        f"Loading observations from {'multiple lines' if multi_line else 'single line'}",
        file=sys.stderr,
    )

    with open(obs_path) as f:
        test_data = [f.read()] if not multi_line else f.readlines()

    ground_obs = [step_from_json(x)["obs"] for x in test_data]

    ground_obs = [filter_obs(o) for o in ground_obs]
    rendered_obs = [render_obs(model, g) for g in ground_obs]

    for x in rendered_obs:
        print(x)


def cli():
    tyro.cli(main)


if __name__ == "__main__":
    cli()
