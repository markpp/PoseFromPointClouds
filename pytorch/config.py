# which dataset to use? 'pre_linemod' "mini_dex" 'sixd_lm' 'tless
data_source = "../datasets/lm"
# number of dims for norm DNN output, 3 => nx, 6 => nx + ny, 9 => nx + ny + nz
out_dim = 9
# which feature extractor to use?
network_type = ["PointNet", "PointNet++", "GNN"][0]
