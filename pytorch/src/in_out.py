import numpy as np
import json
import pandas as pd
from pyntcloud import PyntCloud
import random

def load_x_y(pc_files,n_points=1024): # -1 misses the last one!
    x = []
    y = []
    for pc_file in pc_files:
        # load point cloud
        cloud = PyntCloud.from_file(pc_file)
        pc_data = cloud.points.values
        if pc_data.shape[0] > n_points:
            pose_file = pc_file[:-4]+".json" # -4 or -9
            with open(pose_file, 'r') as file:
                json_data = file.read()
                jps = json.loads(json_data)
                pose = [jps[0]["pos"]["x"], jps[0]["pos"]["y"], jps[0]["pos"]["z"], \
                        jps[0]["orn"]["x"], jps[0]["orn"]["y"], jps[0]["orn"]["z"]]
            x.append(pc_data)
            y.append(pose)
    return np.array(x), np.array(y).astype('float32')

def sample_x(data,n_points=1024):
    candiate_ids = [i for i in range(data.shape[0])]
    sel = []
    for _ in range(n_points):
        # select idx
        idx = random.randint(0,len(candiate_ids)-1)
        sel.append(candiate_ids[idx])
        # remove that idx from point_idx_options
        del candiate_ids[idx]
    return data[sel]

def sample_xs(X,n_points = 1024):
    sample_x = []
    for data in X:
        candiate_ids = [i for i in range(data.shape[0])]
        sel = []
        for _ in range(n_points):
            # select idx
            idx = random.randint(0,len(candiate_ids)-1)
            sel.append(candiate_ids[idx])
            # remove that idx from point_idx_options
            del candiate_ids[idx]
        sample_x.append(data[sel])
    return np.array(sample_x)
