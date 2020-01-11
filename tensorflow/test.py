import os
import numpy as np
import random
import json
from tensorflow.keras.models import load_model
from src.in_out import load_x_y, sample_x

if __name__ == '__main__':
    dataset = 'lay'
    n_points = 1024

    model = load_model("models/{}/model_pn.h5".format(dataset), compile = False)

    for data in ["test","val","train"][:]:
        X = np.load('input/{}/{}_X.npy'.format(dataset,data),allow_pickle=True)
        y = np.load('input/{}/{}_Y.npy'.format(dataset,data),allow_pickle=True).astype('float32')

        x = sample_x(X, n_points)
        pred = model.predict(x)
        pred = np.concatenate((pred[0], pred[1]), axis=1)
        print(pred.shape)
        np.save("input/{}/pred_{}.npy".format(dataset,data),pred)
