import os
import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mean_absolute_error

from src.in_out import DataGenerator, load_x_y, sample_x
from src.model import Points2Pose
from src.schedules import onetenth_150_175
from src.plot import plot_history

from config import data_source, out_dim, batch_size, n_points

def fit_gen(train_gen,val_gen,n_train,n_val,model,epochs=200,batch_size=64,lr=0.01,output_path="models",mode='pn'):

    model.summary()

    opt = Adam(lr=lr)
    if mode == 'pn':
        model.compile(optimizer=opt,
                      loss={'center': 'mean_squared_error', 'norm': 'mean_squared_error'},
                      loss_weights = {"center": 1.0, "norm": 1.0},
                      metrics=['mae'])
    else:
        model.compile(optimizer=opt,
                      loss={'center': 'mean_squared_error'},
                      metrics=['mae'])

    history = model.fit_generator(train_gen.generateXY(mode=mode),
                                  steps_per_epoch=n_train // batch_size,
                                  epochs=epochs,
                                  validation_data=val_gen.generateXY(mode=mode),
                                  validation_steps=n_val // batch_size,
                                  callbacks=[onetenth_150_175(lr)],
                                  verbose=1)

    plot_history([('', history)], output_path, 'model_{}'.format(mode))
    model.save(os.path.join(output_path,"model_{}.h5".format(mode)))

if __name__ == '__main__':

    from_list = False
    if from_list: # read samples from list
        X_val, y_val = load_x_y('{}/val.txt'.format(data_source))
        X_test, y_test = load_x_y('{}/test.txt'.format(data_source))
        X_train, y_train = load_x_y('{}/train.txt'.format(data_source))
    else: # read samples from .npy
        X_train = np.load('{}/{}_X.npy'.format(data_source,"train"),allow_pickle=True)
        y_train = np.load('{}/{}_Y.npy'.format(data_source,"train"),allow_pickle=True)[:,:3+out_dim].astype('float32')
        X_val = np.load('{}/{}_X.npy'.format(data_source,"val"),allow_pickle=True)
        y_val = np.load('{}/{}_Y.npy'.format(data_source,"val"),allow_pickle=True)[:,:3+out_dim].astype('float32')

    #print("Number of samples: {} test".format(X_test.shape[0]))
    print("Number of samples: {} train, {} val".format(X_train.shape[0],X_val.shape[0]))

    model, encoder = Points2Pose(cloud_shape = (n_points, 3), n_latent = 16, out_dim = out_dim)

    # generators that samples n_points from clouds in list and make batches
    train_gen = DataGenerator(X_train, y_train, batch_size=batch_size, augment=True)
    val_gen = DataGenerator(X_val, y_val, batch_size=batch_size, augment=False)
    fit_gen(train_gen, val_gen, 5000, X_val.shape[0], model, output_path="models/{}".format(data_source.split('/')[-1]))
