import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Cropping2D, Conv2D, UpSampling2D, Add, LeakyReLU, Input, Activation, Concatenate
from tensorflow.keras.models import Model
from models.layer_collection import AdaIN, Bias_Layer
import h5py

def Constant_block(shape):
    x = tf.zeros(shape = (shape[0], 4, 4, 512))
    x_out = Bias_Layer() (x)
    return x_out

def A_block(w, filter):
    y_s = Dense(filter)(w)
    y_s = Reshape((1,1,filter))(y_s)
    y_b = Dense(filter)(w)
    y_b = Reshape((1,1,filter))(y_b)
    return y_s, y_b

def B_block(noise, filter, size):
    size = ((0,noise.shape[1] - size),(0,noise.shape[1] - size))
    out = Cropping2D(cropping = size) (noise)
    out = Conv2D(1, (1,1), padding = 'same') (out)
    return out

def G_block(x, w, noise_inp, filter, idx):

    hidden = x
    if idx != 0:
        hidden = UpSampling2D(interpolation = 'bilinear') (hidden)
        hidden = Conv2D(filter,(3,3),padding = 'same') (hidden)

    y_s, y_b = A_block(w, filter+1)
    noise = B_block(noise_inp, filter, hidden.shape[1])
    hidden = Add() ([hidden,noise])
    hidden = AdaIN() ([hidden, y_b, y_s])
    hidden = LeakyReLU(0.2) (hidden)

    hidden = Conv2D(filter,(3,3),padding = 'same') (hidden)
    y_s, y_b = A_block(w, filter+1)
    noise = B_block(noise_inp, filter, hidden.shape[1])
    hidden = Add() ([hidden,noise])
    hidden = AdaIN() ([hidden, y_b, y_s])
    x_out = LeakyReLU(0.2) (hidden)

    to_rgb = Conv2D(3, (1,1), padding = 'same') (x_out)
    to_rgb = UpSampling2D(size = (int(256/hidden.shape[1]),int(256/hidden.shape[1])),interpolation = 'bilinear') (to_rgb)
    return x_out, to_rgb

def mapping_network(latent_size = 512, num_layers = 8):
    z = Input(shape = (latent_size,))

    hidden = z
    for idx in range(num_layers):
        hidden = Dense(latent_size, activation = LeakyReLU(0.2)) (hidden)
    w = hidden

    m = Model(z,w)
    return m

def synthesis_network(configs):
    Feature_maps = [512, 512, 512, 512, 256, 128, 64]
    
    w = Input(shape = (configs['latent_size'],))
    w_shape = tf.shape(w)
    hidden = Constant_block(w_shape)
    noise_inp = Input(shape = (256, 256, 1))
    x_out = []

    for idx in range(len(Feature_maps)):
        hidden, rgb = G_block(hidden, w, noise_inp, Feature_maps[idx], idx)
        x_out.append(rgb)

    x_out = Add() (x_out)
    x_out = Activation('tanh') (x_out)
    model = Model([w,noise_inp],x_out)
    return model

def get_generator(configs):
    F = mapping_network(latent_size=configs['latent_size'], num_layers=configs['mapping_size'])
    G = synthesis_network(configs)

    return F, G

def get_FG(configs, F, G):
    H, W = configs['image_height'], configs['image_width']
    latent_size = configs['latent_size']

    z = Input(shape = (latent_size,))
    noise_inp = Input(shape = (H,W,1))
    
    m = Model([z,noise_inp], G([F(z),noise_inp]))
    return m


def check_point_G(Generator, path):
    F, G = Generator
    

    


