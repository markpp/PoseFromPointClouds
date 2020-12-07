import torch
import json
import argparse
import numpy as np
import os
from itertools import islice
import yaml
#import open3d as o3d
from pyntcloud import PyntCloud
import random
from config import data_source

def load_x_y(list_path, n_points=1024):
    X, Y = [], []
    with open(list_path) as f:
        pc_files = f.read().splitlines()[:]
        #random.shuffle(pc_files)

    for pc_file in pc_files:
        # load point cloud
        cloud = PyntCloud.from_file(pc_file)
        pc_data = cloud.points.values[:,:3]
        if pc_data.shape[0] > n_points:
            pose_file = pc_file[:-4]+".json" # -4 or -9
            with open(pose_file, 'r') as file:
                json_data = file.read()
                jps = json.loads(json_data)
                pose = [jps[0]["pos"]["x"], jps[0]["pos"]["y"], jps[0]["pos"]["z"], \
                        jps[0]["nx"]["x"], jps[0]["nx"]["y"], jps[0]["nx"]["z"]]#, \
                        #jps[0]["ny"]["x"], jps[0]["ny"]["y"], jps[0]["ny"]["z"], \
                        #jps[0]["nz"]["x"], jps[0]["nz"]["y"], jps[0]["nz"]["z"]]
            if pc_data.shape[0] > 1024:
                X.append(pc_data[:,:])
                Y.append(pose)
            else:
                print(pc_data.shape[0])
    return np.array(X), np.array(Y), pc_files

def move_to_origo(points, center_mass=True):
    # find center
    if center_mass:
        xmean = np.mean(points[:,0])
        ymean = np.mean(points[:,1])
        zmean = np.mean(points[:,2])
        center = [xmean,ymean,zmean]
    else:
        xmin, xmax = np.min(points[:,0]), np.max(points[:,0])
        ymin, ymax = np.min(points[:,1]), np.max(points[:,1])
        zmin, zmax = np.min(points[:,2]), np.max(points[:,2])
        center = [xmin+(xmax-xmin)/2,ymin+(ymax-ymin)/2,zmin+(zmax-zmin)/2]
    # translate points to 0,0,0
    points[:,0] -= center[0]
    points[:,1] -= center[1]
    points[:,2] -= center[2]
    return points, center

def sample_N_random(x,n_points=1024):
    candiate_ids = [i for i in range(x.shape[0])]
    sel = []
    for _ in range(n_points):
        # select idx
        idx = random.randint(0,len(candiate_ids)-1)
        sel.append(candiate_ids[idx])
        # remove that idx from point_idx_options
        del candiate_ids[idx]
    return np.array(x[sel])

if __name__ == '__main__':
    """
    Main function for executing the .py script.
    Command:
        -l val.txt (add -m if you want mirror)
    """
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--list", type=str,
                    help="list of pc files")
    args = vars(ap.parse_args())
    
    X, Y, paths = load_x_y(args["list"])

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("models/best_model.pkl")
    model.eval()
    model = model.to(dev)


    with torch.no_grad():
        for x, y, path in zip(X,Y,paths):
            x, offset = move_to_origo(sample_N_random(x))
            x = torch.from_numpy(np.expand_dims(x, axis=0))
            x = x.to(dev)
            pred_p, feat_p, pred_n, feat = model(x)
            pred = torch.cat([pred_p, pred_n], 1)
            if dev.type == 'cuda':
                pred = pred.cpu()
                feat = feat.cpu()
            pred = pred.data.numpy()[0]
            feat = feat.data.numpy()[0]
            t, nx = (pred[:3]+offset), pred[3:6]
            
            # load image
            print(path)

            break