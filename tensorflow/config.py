# which dataset to use? 'pre_linemod' "dex"
data_source = "../datasets/lm"
# number of dims for norm DNN output, 3 => nx, 6 => nx + ny, 9 => nx + ny + nz
out_dim = 9
# which feature extractor to use?
network_type = ["PointNet", "PointNet++", "GNN"][0]
batch_size = 32
n_points = 1024

