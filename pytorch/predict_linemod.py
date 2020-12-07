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

def read_yml_gt(path):
    with open(path) as gt:
        instance_idxs, p0s, rots = [], [], []
        documents = yaml.full_load(gt)
        for instance_idx, content in documents.items():
            instance_idxs.append(int(instance_idx))
            p0s.append(np.array(content[0]['cam_t_m2c'])/1000.0)
            rot = content[0]['cam_R_m2c']
            rots.append(np.array([[rot[0], rot[1], rot[2]],
                                    [rot[3], rot[4], rot[5]],
                                    [rot[6], rot[7], rot[8]]]))
        return instance_idxs, p0s, rots

def read_json_gt(path):
    with open(path) as gt:
        instance_idxs, p0s, rots = [], [], []
        data = json.load(gt)
        for instance_idx in data:
            instance_idxs.append(int(instance_idx))
            p0s.append(np.array(data[instance_idx][0]['cam_t_m2c'])/1000.0)
            rot = data[instance_idx][0]['cam_R_m2c']
            rots.append(np.array([[rot[0], rot[1], rot[2]],
                                    [rot[3], rot[4], rot[5]],
                                    [rot[6], rot[7], rot[8]]]))
        return instance_idxs, p0s, rots

def write_yml_pred(path):
    with open(path, 'w') as f:
        f.write("run_time: -1\n")
        f.write("ests:\n")
        f.write("- {{score: {}, ".format(100.0))
        f.write("R: [{}, {}, {}, {}, {}, {}, {}, {}, {}], ".format(
                    nx[0],ny[0],nz[0],
                    nx[1],ny[1],nz[1],
                    nx[2],ny[2],nz[2]))
        f.write("t: [{},{},{}]}}".format(t[0],t[1],t[2]))

def write_json_pred(path,p0,nx,ny,nz):
    with open(path, 'w') as f:
        jo = [{
            "pos": {
                "x": float(p0[0]),
                "y": float(p0[1]),
                "z": float(p0[2])
            },
            "nx": {
                "x": float(nx[0]),
                "y": float(nx[1]),
                "z": float(nx[2])
            },
            "ny": {
                "x": float(ny[0]),
                "y": float(ny[1]),
                "z": float(ny[2])
            },
            "nz": {
                "x": float(nz[0]),
                "y": float(nz[1]),
                "z": float(nz[2])
            }
        }]
        f.write(json.dumps(jo, indent=2, sort_keys=False))

if __name__ == '__main__':
    dataset = 'data'
    in_path = '{}/{}'.format(data_source,dataset) #'sixd_hinterstoisser/train'
    #camera_intr = CameraIntrinsics()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("models/final_model.pkl")
    model.eval()
    model = model.to(dev)

    csv_file = open('{}/results/mark-sensors-mask_lm-{}.csv'.format(data_source,dataset),'w')
    csv_file.write("scene_id,im_id,obj_id,score,R,t,time\n")

    with torch.no_grad():
        for object_idx in range(1,15+1)[:1]:
            object_idx = str(object_idx).zfill(6)
            print("Doing object {}".format(object_idx))
            instance_idxs, p0s, rots = read_json_gt('{}/{}/scene_gt.json'.format(in_path,object_idx))

            for idx,instance_idx in enumerate(instance_idxs[:]):
                filename = str(instance_idx).zfill(6)
                cloud = PyntCloud.from_file("{}/{}/points/{}.ply".format(in_path,object_idx,filename))
                object_points = cloud.points.values[:,:3].astype('float32')

                x, offset = move_to_origo(sample_N_random(object_points))
                x = torch.from_numpy(np.expand_dims(x, axis=0))
                x = x.to(dev)
                pred_p, feat_p, pred_n, feat = model(x)
                pred = torch.cat([pred_p, pred_n], 1)
                if dev.type == 'cuda':
                    pred = pred.cpu()
                    feat = feat.cpu()
                pred = pred.data.numpy()[0]
                feat = feat.data.numpy()[0]
                t, nx, ny, nz = (pred[:3]+offset), pred[3:6], pred[6:9], pred[9:12]
                #nx = (pred[3:6]-pred[:3])*10.0
                nx = nx / np.linalg.norm(nx)
                ny = ny / np.linalg.norm(ny)
                nz = nz / np.linalg.norm(nz)

                #nz = np.cross(ny,nx)

                #print("pred: {}, {}".format(t,ns))
                #print("gt: {}, {}".format(p0s[idx],rots[idx]))
                #pred_yml_pred("{}/gt/{}_{}.yml".format(in_path,filename,object_idx))
                write_json_pred("{}/{}/points/{}_.json".format(in_path,object_idx,filename),t,nx,ny,nz)

                csv_file.write("{},{},{},{},".format(int(object_idx),int(instance_idx),int(object_idx),1.0))
                csv_file.write("{:.6f} {:.6f} {:.6f} ".format(nx[0],ny[0],nz[0]))
                csv_file.write("{:.6f} {:.6f} {:.6f} ".format(nx[1],ny[1],nz[1]))
                csv_file.write("{:.6f} {:.6f} {:.6f},".format(nx[2],ny[2],nz[2]))
                csv_file.write("{:.6f} {:.6f} {:.6f}, 2.0\n".format(t[0]*1000.0,t[1]*1000.0,t[2]*1000.0))


    csv_file.close()
    #https://github.com/sjtuytc/segmentation-driven-pose/blob/master/evaluate.py
    #https://github.com/thodan/sixd_toolkit/blob/96bb268e1fb5ebd82ca1b8d352e3263561ba6f5c/pysixd/pose_error.py#L154
