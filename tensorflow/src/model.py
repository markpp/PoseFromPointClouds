from tensorflow.keras.layers import Layer, Conv1D, MaxPooling1D, Flatten, Dropout, Input, BatchNormalization, Dense
from tensorflow.keras.layers import Reshape, Lambda, concatenate
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf

# PointNet without input and feature transform sub networks
def PointNet(input_points):
    g = Conv1D(64, 1, activation='relu')(input_points)
    g = BatchNormalization()(g)
    g = Conv1D(64, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Conv1D(128, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Conv1D(1024, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    global_feature = MaxPooling1D(pool_size=1024)(g)
    return global_feature
'''
def Encoder(feature, n_latent, name="enc"):
    e = Dense(512, activation='relu')(feature)
    e = BatchNormalization()(e)
    e = Dropout(0.5)(e)
    e = Dense(256, activation='relu')(e)
    e = BatchNormalization()(e)
    e = Dropout(0.5)(e)
    e = Dense(n_latent, activation='linear')(e)
    z = Flatten(name='{}_z'.format(name))(e)
    return z

def Center(latent):
    c = Dense(48, activation='relu', name='c0')(latent)
    c = BatchNormalization(name='cbn0')(c)
    c = Dropout(0.5)(c)
    c = Dense(24, activation='relu', name='c1')(c)
    c = BatchNormalization(name='cbn1')(c)
    c = Dropout(0.5)(c)
    c = Dense(8, activation='relu', name='c2')(c)
    c = Dense(3, activation='linear', name='c3')(c)
    c = Flatten(name='center')(c)
    return c

def Normal(latent):
    n = Dense(48, activation='relu', name='n0')(latent)
    n = Dense(24, activation='relu', name='n1')(n)
    n = Dense(8, activation='relu', name='n2')(n)
    n = Dense(3, activation='linear', name='n3')(n)
    n = Flatten(name='norm')(n)
    return n

def Points2Point(cloud_shape, n_latent):
    input_points = Input(shape=cloud_shape)

    # create PointNet model
    cloud_feature = PointNet(input_points)
    # create Encoder model
    z = Encoder(cloud_feature, n_latent)
    encoder = Model(inputs=input_points, outputs=z)
    # center point regressor
    center = Center(z)
    model = Model(inputs=input_points, outputs=center)

    #norm = Normal(z)
    #model = Model(inputs=input_points, outputs=[center, norm])
    return model, encoder
'''


def Encoder(feature, n_latent, name="enc"):
    e = Dense(512, activation='relu', name='{}_c0'.format(name))(feature)
    e = BatchNormalization(name='{}_bn0'.format(name))(e)
    e = Dropout(0.5)(e)
    e = Dense(256, activation='relu', name='{}_c1'.format(name))(e)
    e = BatchNormalization(name='{}_bn1'.format(name))(e)
    e = Dropout(0.25)(e)
    e = Dense(n_latent, activation='linear', name='{}_c2'.format(name))(e)
    e = Flatten(name='{}_e'.format(name))(e)
    return e

def PoseNet(latent, name="pn", out_dim=3):
    p = Dense(48, activation='relu', name='{}_c0'.format(name))(latent)
    p = BatchNormalization(name='{}_bn0'.format(name))(p)
    #p = Dropout(0.25)(p)
    p = Dense(24, activation='relu', name='{}_c1'.format(name))(p)
    p = BatchNormalization(name='{}_bn1'.format(name))(p)
    #p = Dropout(0.5)(p)
    p = Dense(8, activation='relu', name='{}_c2'.format(name))(p)
    p = Dense(out_dim, activation='linear', name='{}_c3'.format(name))(p)
    p = Flatten(name=name)(p)
    return p

def Points2Pose(cloud_shape, n_latent, out_dim):
    input_points = Input(shape=cloud_shape)
    # create PointNet model
    cloud_feature_n = PointNet(input_points)
    cloud_feature_c = PointNet(input_points)
    # create Encoder model
    z_n = Encoder(cloud_feature_n, n_latent, "n_enc")
    encoder_n = Model(inputs=input_points, outputs=z_n)
    z_c = Encoder(cloud_feature_c, n_latent, "c_enc")
    encoder_c = Model(inputs=input_points, outputs=z_c)
    # center point regressor
    center = PoseNet(z_c, "center")
    # normal regressor
    norm = PoseNet(z_n, "norm", out_dim)
    model = Model(inputs=input_points, outputs=[center, norm])
    return model, encoder_n#, encoder_c
