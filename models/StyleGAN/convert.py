from utils.plotlib import display_img
from keras.models import Model
from keras.layers import Conv2D, Dense, Activation, Flatten, Reshape, Input, UpSampling2D, Add
from keras.layers import LeakyReLU, Cropping2D, AveragePooling2D
from keras.optimizers import Adam
from Custom_layers import AdaIN
import tensorflow as tf
import numpy as np
from StyleGAN_model import replicate,A_block,B_block,D_block,G_block
import os

img_height, img_width = 256,256

path = "trained_model"
const = np.loadtxt(os.path.join(path, 'key_const.bin'),delimiter=',')
const = np.reshape(const,(4,4,512))
const_tensor = replicate(const, 4)
cnt = 0

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
        hidden = AdaIN() ([hidden, y_b, y_s])
        hidden = Activation('relu') (hidden)

        hidden, rgb = G_block(hidden, w, noise_inp, 512)
        x_out.append(rgb)
        hidden, rgb = G_block(hidden, w, noise_inp, 256)
        x_out.append(rgb)
        hidden, rgb = G_block(hidden, w, noise_inp, 128)
        x_out.append(rgb)
        hidden, rgb = G_block(hidden, w, noise_inp, 64)
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
        hidden = AdaIN() ([hidden, y_b, y_s])
        hidden = Activation('relu') (hidden)

        hidden, rgb = G_block(hidden, w, noise_inp, 512)
        x_out.append(rgb)
        hidden, rgb = G_block(hidden, w, noise_inp, 256)
        x_out.append(rgb)
        hidden, rgb = G_block(hidden, w, noise_inp, 128)
        x_out.append(rgb)
        hidden, rgb = G_block(hidden, w, noise_inp, 64)
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
