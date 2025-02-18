import graphviz

logfile = "test_imitation_mdp_render.log"

frame_folder = "frames"

with open(logfile, "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    graph = graphviz.Source(line)
    graph.render(f"{frame_folder}/frame_{i}", format="jpg", engine="sfdp")
