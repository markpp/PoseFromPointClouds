# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Configuration of the BOP Toolkit."""

######## Basic ########
dataset = 'lm' #'pre_linemod'

# Folder with the BOP datasets.
datasets_path = r'/home/markpp/github/PoseFromPointClouds/datasets'

# Folder with pose results to be evaluated.
results_path = r'/home/markpp/github/PoseFromPointClouds/datasets/{}/results'.format(dataset)

# Folder for the calculated pose errors and performance scores.
eval_path = r'/home/markpp/github/PoseFromPointClouds/datasets/{}/eval'.format(dataset)

######## Extended ########

# Folder for outputs (e.g. visualizations).
output_path = r'/home/markpp/github/PoseFromPointClouds/datasets/{}/vis'.format(dataset)

# For offscreen C++ rendering: Path to the build folder of bop_renderer (github.com/thodan/bop_renderer).
bop_renderer_path = r'/path/to/bop_renderer/build'

# Executable of the MeshLab server.
meshlab_server_path = r'/path/to/meshlabserver.exe'
