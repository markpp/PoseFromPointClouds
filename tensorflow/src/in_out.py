import numpy as np
import random
import pandas as pd
from pyntcloud import PyntCloud
import os
import json
import math

from tensorflow.keras import models

#if not os.path.exists('./results/'):
#    os.mkdir('./results/')
#if not os.path.exists('./models/'):
#    os.mkdir('./models/')
#if not os.path.exists('./output/'):
#    os.mkdir('./output/')

def load_x_y(list_path, n_points = 1024):
    x = []
    y = []
    with open(list_path) as f:
        pc_files = f.read().splitlines()
        random.shuffle(pc_files)

    for pc_file in pc_files[:]:
        # load point cloud
        cloud = PyntCloud.from_file(pc_file)
        pc_data = cloud.points.values[:,:3]
        if pc_data.shape[0] > n_points:
            pose_file = pc_file[:-4]+".json" # -4 or -9
            with open(pose_file, 'r') as file:
                json_data = file.read()
                jps = json.loads(json_data)
                pose = [jps[0]["pos"]["x"], jps[0]["pos"]["y"], jps[0]["pos"]["z"], \
                        jps[0]["orn"]["x"], jps[0]["orn"]["y"], jps[0]["orn"]["z"]]
            x.append(pc_data)
            y.append(pose)
    return np.array(x), np.array(y)

def load_x(list_path, n_points = 1024, n_samples = -1):
    x = []
    with open(list_path) as f:
        pc_files = f.read().splitlines()
    for pc_file in pc_files[:n_samples]:
        # load point cloud
        cloud = PyntCloud.from_file(pc_file)
        pc_data = cloud.points.values
        if pc_data.shape[0] > n_points:
            x.append(pc_data)
    return np.array(x)

def sample_x(X, n_points = 1024):
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


class DataGenerator:
    def __init__(self, x, y = None, batch_size = 64, n_points = 1024, augment = True, model_path = None):
        self.x = x
        self.y = y
        self.n_samples = x.shape[0]
        self.batch_size = batch_size
        self.n_points = n_points
        self.augment = augment
        if model_path is not None:
            self.model = models.load_model(model_path, compile=False)
            self.model._make_predict_function()

    @staticmethod
    def sample_point_cloud(data, n_points):
        """ Randomly sample the required number of points from the point cloud
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, sampled point clouds
        """
        candiate_ids = [i for i in range(data.shape[0])]
        sel = []
        for _ in range(n_points):
            # select idx
            idx = random.randint(0,len(candiate_ids)-1)
            sel.append(candiate_ids[idx])
            # remove that idx from point_idx_options
            del candiate_ids[idx]
        return data[sel]

    @staticmethod
    def rotate_point_cloud(data, rotation_angle):
        """ Randomly rotate the point clouds to augument the dataset
            rotation is per shape based along up direction
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, rotated point clouds
        """
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)

        # y
        #rotation_matrix = np.array([[cosval, 0, sinval],[0, 1, 0],[-sinval, 0, cosval]])
        # z
        rotation_matrix = np.array([[cosval, -sinval, 0],[sinval, cosval, 0],[0, 0, 1]])
        # debug
        #rotation_matrix = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])

        rotated_data = np.dot(data.reshape((-1, 3)), rotation_matrix)
        return rotated_data

    @staticmethod
    def scale_point_cloud(data, min=0.9, max=1.1):
        """ Randomly scale points. scaling is per cloud.
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, scaled point clouds
        """
        return data * random.uniform(min, max)

    @staticmethod
    def jitter_point_cloud(data, sigma=0.005, clip=0.01):
        """ Randomly jitter points. jittering is per point and per cloud.
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, jittered point clouds
        """
        N, C = data.shape
        assert (clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip) + random.uniform(-0.02, 0.02)
        jittered_data += data
        return jittered_data

    def generateXY(self, mode):
        while True:
            index = [n for n in range(self.n_samples)]
            random.shuffle(index)
            for i in range(self.n_samples // self.batch_size):
                batch_start, batch_end = i * self.batch_size, (i + 1) * self.batch_size
                batch_index = index[batch_start: batch_end]
                batch_x, batch_y_center, batch_y_norm = [], [], []
                for j in batch_index:
                    sample_x = self.sample_point_cloud(self.x[j], self.n_points)
                    sample_y = self.y[j]
                    if self.augment:
                        is_rotate = 0 #random.randint(0, 1)
                        is_scale = 0
                        is_jitter = 0
                        if is_rotate == 1:
                            rotation_angle = np.random.uniform(0.0,2*math.pi)
                            sample_x = self.rotate_point_cloud(sample_x, rotation_angle)
                            p0, pn = self.rotate_point_cloud(np.array([sample_y[:3],sample_y[:3]+sample_y[3:]]), rotation_angle)
                            sample_y = np.concatenate((p0, pn-p0), axis=0)
                        if is_scale == 1:
                            sample_x = self.scale_point_cloud(sample_x)
                        if is_jitter == 1:
                            sample_x = self.jitter_point_cloud(sample_x)
                    batch_x.append(sample_x)
                    if mode == 'p':
                        batch_y_center.append(sample_y[:3])
                    elif mode == 'n':
                        batch_y_center.append(sample_y[3:])
                    else:
                        batch_y_center.append(sample_y[:3])
                        batch_y_norm.append(sample_y[3:])

                #
                if mode == 'pn':
                    yield np.array(batch_x), {'center': np.array(batch_y_center), 'norm': np.array(batch_y_norm)}
                else:
                    yield np.array(batch_x), {'center': np.array(batch_y_center)}


    def generateEncY(self):
        while True:
            index = [n for n in range(self.n_samples)]
            random.shuffle(index)
            for i in range(self.n_samples // self.batch_size):
                batch_start, batch_end = i * self.batch_size, (i + 1) * self.batch_size
                batch_index = index[batch_start: batch_end]
                batch_x, batch_y_center, batch_y_norm = [], [], []
                for j in batch_index:
                    sample_x = self.sample_point_cloud(self.x[j], self.n_points)
                    sample_y = self.y[j]
                    if self.augment:
                        is_rotate = 0 #random.randint(0, 1)
                        is_jitter = 0
                        if is_rotate == 1:
                            rotation_angle = np.random.uniform(0.0,2*math.pi)
                            sample_x = self.rotate_point_cloud(sample_x, rotation_angle)
                            p0, pn = self.rotate_point_cloud(np.array([sample_y[:3],sample_y[:3]+sample_y[3:]]), rotation_angle)
                            sample_y = np.concatenate((p0, pn-p0), axis=0)
                        if is_jitter == 1:
                            sample_x = self.jitter_point_cloud(sample_x)
                    batch_x.append(sample_x)
                    batch_y_center.append(sample_y[:3])
                    batch_y_norm.append(sample_y[3:])
                batch_enc = self.model.predict(np.array(batch_x))
                yield batch_enc, {'center': np.array(batch_y_center), 'norm': np.array(batch_y_norm)}

if __name__ == '__main__':
    # inspect the generator output
    n_examples = 4

    # load dataset
    val_X, val_Y = load_x_and_y('valx.txt', n_examples)

    # baseline sampling(no augmentation)
    val_x = sample_x(val_X, 1024)

    # sample data generator
    val_gen = DataGenerator(val_X, val_Y, n_examples, augment=True)
    gen_x, gen_y = next(val_gen.generateXY())
    gen_y = np.concatenate((gen_y['center'], gen_y['norm']), axis=1)

    for idx, x in enumerate(gen_x):
        points2file(val_X[idx],"output/val_{}_crop.ply".format(idx))
        pose2json(val_Y[idx][:3], val_Y[idx][3:],"output/val_{}_crop.json".format(idx))

        sample_x = val_gen.sample_point_cloud(val_X[idx], val_gen.n_points)
        points2file(sample_x,"output/val_{}_sample.ply".format(idx))
        pose2json(val_Y[idx][:3], val_Y[idx][3:],"output/val_{}_sample.json".format(idx))

        #points2obj(val_x[idx],"output/val_x_{}.obj".format(idx))
        #anno2obj(val_Y[idx][:3], val_Y[idx][3:],"output/val_y_{}.obj".format(idx))
        #points2obj(x,"output/gen_x_{}.obj".format(idx))
        #anno2obj(gen_y[idx][:3], gen_y[idx][3:],"output/gen_y_{}.obj".format(idx))
