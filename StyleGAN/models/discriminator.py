import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Cropping2D, Conv2D, UpSampling2D, Add, LeakyReLU, AveragePooling2D, Input, Flatten
from tensorflow.keras.models import Model

from models.layer_collection import Minibatch_stddev

def D_block(x, filter, idx):

    if idx == 0:
        x = Conv2D(filter,(1,1),padding='same') (x)
        x = LeakyReLU(0.2) (x)

    hidden = Conv2D(filter,(3,3),padding = 'same') (x)
    hidden = LeakyReLU(0.2) (hidden)
    hidden = Conv2D(filter,(3,3),padding = 'same') (hidden)
    hidden = LeakyReLU(0.2) (hidden)

    x_out = Conv2D(filter, (1,1), padding = 'same') (x)
    x_out = Add()([x_out,hidden])
    x_out = AveragePooling2D() (x_out)
    return x_out

def get_discriminator():
    Feature_maps = [64, 128, 256, 512, 512, 512]

    x = Input(shape = (256, 256, 3))
    hidden = x
    
    for idx in range(len(Feature_maps)):
        hidden = D_block(hidden, Feature_maps[idx], idx)

    hidden = Minibatch_stddev() (hidden)
    hidden = Conv2D(Feature_maps[-1], (3,3), padding = 'same') (hidden)
    hidden = LeakyReLU(0.2) (hidden)
    hidden = Conv2D(Feature_maps[-1], (4,4)) (hidden)
    hidden = LeakyReLU(0.2) (hidden)

    hidden = Flatten() (hidden)
    x_out = Dense(1, activation='linear') (hidden)
    model = Model(x,x_out)
    return model