from utils.plotlib import display_img
from keras.models import Model
from keras.layers import Conv2D, Dense, Activation, Flatten, Reshape, Input, UpSampling2D, Add
from keras.layers import LeakyReLU, Cropping2D, AveragePooling2D
from keras.optimizers import Adam
from AdaIN import AdaInstanceNormalization
import tensorflow as tf
import numpy as np
from StyleGAN_model import replicate
import os

img_height, img_width = 256,256

path = "trained_model"
const = np.loadtxt(os.path.join(path, 'key_const.bin'),delimiter=',')
const = np.reshape(const,(4,4,512))
const_tensor = replicate(const, 4)
cnt = 0

def A_block(w, filter):
    y_s = Dense(filter)(w)
    y_s = Reshape((1,1,filter))(y_s)
    y_b = Dense(filter)(w)
    y_b = Reshape((1,1,filter))(y_b)
    return y_s, y_b

def B_block(noise, filter, size):
    size = ((0,noise.shape[1] - size),(0,noise.shape[1] - size))
    out = Cropping2D(cropping = size) (noise)
    out = Conv2D(filter, (1,1), padding = 'same') (out)
    return out

def G_block(x, w, noise_inp, filter):
    hidden = UpSampling2D() (x)

    hidden = Conv2D(filter,(3,3),padding = 'same') (hidden)
    noise = B_block(noise_inp, filter, hidden.shape[1])
    y_s, y_b = A_block(w, filter)
    hidden = Add() ([hidden,noise])
    hidden = AdaInstanceNormalization() ([hidden, y_b, y_s])
    hidden = Activation('relu') (hidden)

    hidden = Conv2D(filter,(3,3),padding = 'same') (hidden)
    noise = B_block(noise_inp, filter, hidden.shape[1])
    y_s, y_b = A_block(w, filter)
    hidden = Add() ([hidden,noise])
    hidden = AdaInstanceNormalization() ([hidden, y_b, y_s])
    x_out = Activation('relu') (hidden)

    to_rgb = Conv2D(3, (1,1), padding = 'same') (x_out)
    to_rgb = UpSampling2D(size = (int(256/hidden.shape[1]),int(256/hidden.shape[1]))) (to_rgb)
    return x_out, to_rgb
    
def D_block(x, filter):
    hidden = Conv2D(filter,(3,3),padding = 'same') (x)
    hidden = LeakyReLU(0.2) (hidden)
    hidden = Conv2D(filter,(3,3),padding = 'same') (hidden)
    hidden = LeakyReLU(0.2) (hidden)

    x_out = Conv2D(filter, (1,1), padding = 'same') (x)
    x_out = Add()([x_out,hidden])
    x_out = AveragePooling2D() (x_out)
    return x_out

def Generator_old():
        x_out = []
        
        z = Input(shape = (512,))
        FC = Dense(512, activation = 'relu',name = 'FC_1') (z)
        FC = Dense(512, activation = 'relu',name = 'FC_2') (FC)
        FC = Dense(512, activation = 'relu',name = 'FC_3') (FC)
        FC = Dense(512, activation = 'relu',name = 'FC_4') (FC)
        FC = Dense(512, activation = 'relu',name = 'FC_5') (FC)
        FC = Dense(512, activation = 'relu',name = 'FC_6') (FC)
        FC = Dense(512, activation = 'relu',name = 'FC_7') (FC)
        w = Dense(512, activation = 'relu',name = 'FC_8') (FC)
        noise_inp = Input(shape = (img_height,img_width,1))

        x = const_tensor
        y_s, y_b = A_block(w, 512)
        noise = B_block(noise_inp, 512, 4)
        hidden = Add() ([x,noise])
        hidden = AdaInstanceNormalization() ([hidden, y_b, y_s])
        hidden = Activation('relu') (hidden)
        hidden, rgb = G_block(hidden, w, noise_inp, 512)
        x_out.append(rgb)
        hidden, rgb = G_block(hidden, w, noise_inp, 256)
        x_out.append(rgb)
        hidden, rgb = G_block(hidden, w, noise_inp, 128)
        x_out.append(rgb)
        hidden, rgb = G_block(hidden, w, noise_inp, 64)
        x_out.append(rgb)
        hidden, rgb = G_block(hidden, w, noise_inp, 32)
        x_out.append(rgb)
        hidden, rgb = G_block(hidden, w, noise_inp, 16)
        x_out.append(rgb)
        x_out = Add() (x_out)
        x_out = Activation('tanh') (x_out)
        model = Model([z,noise_inp],x_out)
        return model


def Mapping_network():
        z = Input(shape = (512,))
        FC = Dense(512, activation = 'relu',name = 'FC_1') (z)
        FC = Dense(512, activation = 'relu',name = 'FC_2') (FC)
        FC = Dense(512, activation = 'relu',name = 'FC_3') (FC)
        FC = Dense(512, activation = 'relu',name = 'FC_4') (FC)
        FC = Dense(512, activation = 'relu',name = 'FC_5') (FC)
        FC = Dense(512, activation = 'relu',name = 'FC_6') (FC)
        FC = Dense(512, activation = 'relu',name = 'FC_7') (FC)
        w = Dense(512, activation = 'relu',name = 'FC_8') (FC)
        model = Model(z,w)
        return model

def Generator():
        x_out = []
        
        
        w = Input(shape = (512,))
        noise_inp = Input(shape = (img_height,img_width,1))

        x = const_tensor
        y_s, y_b = A_block(w, 512)
        noise = B_block(noise_inp, 512, 4)
        hidden = Add() ([x,noise])
        hidden = AdaInstanceNormalization() ([hidden, y_b, y_s])
        hidden = Activation('relu') (hidden)
        hidden, rgb = G_block(hidden, w, noise_inp, 512)
        x_out.append(rgb)
        hidden, rgb = G_block(hidden, w, noise_inp, 256)
        x_out.append(rgb)
        hidden, rgb = G_block(hidden, w, noise_inp, 128)
        x_out.append(rgb)
        hidden, rgb = G_block(hidden, w, noise_inp, 64)
        x_out.append(rgb)
        hidden, rgb = G_block(hidden, w, noise_inp, 32)
        x_out.append(rgb)
        hidden, rgb = G_block(hidden, w, noise_inp, 16)
        x_out.append(rgb)
        x_out = Add() (x_out)
        x_out = Activation('tanh') (x_out)
        model = Model([w,noise_inp],x_out)
        return model

cnt = 0
old_gen = Generator_old()
old_gen.load_weights("trained_model/old_Generator.h5")
for layer in old_gen.layers:
        if layer.name[0] != 'F':
                cnt+=1
                layer._name = str(cnt)


cnt = 0
map_net = Mapping_network()
synthesis = Generator()


for layer in synthesis.layers:
        cnt+=1
        layer._name = str(cnt)
        layer.set_weights(old_gen.get_layer(name = layer.name).get_weights())


for layer in map_net.layers:
        if layer.name[0] == 'F':
                layer.set_weights(old_gen.get_layer(name = layer.name).get_weights())


map_net.save_weights("trained_model/Mapping_Network.h5")
synthesis.save_weights("trained_model/Generator.h5")
