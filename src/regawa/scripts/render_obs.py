
from regawa.io import step_from_json
from regawa.model.base_grounded_model import GroundObs
from regawa.model.base_model import BaseModel
from regawa.model.utils import model_from_json
from regawa.wrappers.graph_utils import fn_groundobs_to_heterograph
from regawa.wrappers.render_utils import create_render_graph, to_graphviz
from pathlib import Path
import tyro

#test_model_path = "tyrLang_regawa_model.json"
#test_obs_path = "single_obs.json"




def render_obs(model: BaseModel, obs: GroundObs):
	obs_to_graph = fn_groundobs_to_heterograph(model, False)
	hetero_graph = obs_to_graph(obs)
	render_graph = create_render_graph(hetero_graph.boolean, hetero_graph.numeric)
	return to_graphviz(render_graph, pprint=True)


def main(test_model_path: Path, test_obs_path: Path):
	with open(test_model_path, "r") as f:
		model_json = f.read()
	model = model_from_json(model_json)

	with open(test_obs_path, "r") as f:
		test_data = f.read()

	ground_obs = step_from_json(test_data)['obs']

	rendered_obs = render_obs(model, ground_obs)
	print(rendered_obs)

def cli():
	tyro.cli(main)

if __name__ == "__main__":