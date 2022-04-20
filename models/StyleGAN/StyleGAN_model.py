import pickle
from tkinter import Image
from loss_function import *
from keras.models import Model
from keras.layers import Conv2D, Dense, Activation, Flatten, Reshape, Input, UpSampling2D, Add
from keras.layers import LeakyReLU, Cropping2D, AveragePooling2D
import tensorflow as tf
from keras.optimizers import Adam
from Custom_layers import AdaIN

import h5py

import os


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
    hidden = UpSampling2D(interpolation = 'bilinear') (x)

    hidden = Conv2D(filter,(3,3),padding = 'same') (hidden)
    noise = B_block(noise_inp, filter, hidden.shape[1])
    y_s, y_b = A_block(w, filter)
    hidden = Add() ([hidden,noise])
    hidden = AdaIN() ([hidden, y_b, y_s])
    hidden = LeakyReLU(0.2) (hidden)

    hidden = Conv2D(filter,(3,3),padding = 'same') (hidden)
    noise = B_block(noise_inp, filter, hidden.shape[1])
    y_s, y_b = A_block(w, filter)
    hidden = Add() ([hidden,noise])
    hidden = AdaIN() ([hidden, y_b, y_s])
    x_out = LeakyReLU(0.2) (hidden)

    to_rgb = Conv2D(3, (1,1), padding = 'same') (x_out)
    to_rgb = UpSampling2D(size = (int(256/hidden.shape[1]),int(256/hidden.shape[1])),interpolation = 'bilinear') (to_rgb)
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

def replicate(const, batch_size):
    ans = []
    for _ in range(batch_size):
        ans.append(const)
    ans = tf.constant(np.array(ans), dtype = float)
    return ans

class StyleGAN():
    
    def _Get_Mapping_Network(self):
        z = Input(shape = (512,))
        FC = Dense(512) (z)
        FC = LeakyReLU(0.2) (FC)
        FC = Dense(512) (FC)
        FC = LeakyReLU(0.2) (FC)
        FC = Dense(512) (FC)
        FC = LeakyReLU(0.2) (FC)
        FC = Dense(512) (FC)
        FC = LeakyReLU(0.2) (FC)
        FC = Dense(512) (FC)
        FC = LeakyReLU(0.2) (FC)
        FC = Dense(512) (FC)
        FC = LeakyReLU(0.2) (FC)
        FC = Dense(512) (FC)
        FC = LeakyReLU(0.2) (FC)
        FC = Dense(512) (FC)
        w = LeakyReLU(0.2) (FC)
        return Model(z,w)

    def _Get_Generator(self):
        x_out = []
        
        
        w = Input(shape = (512,))
        noise_inp = Input(shape = (self.img_height,self.img_width,1))

        x = self.const_tensor
        y_s, y_b = A_block(w, 512)
        noise = B_block(noise_inp, 512, 4)
        hidden = Add() ([x,noise])
        hidden = AdaIN() ([hidden, y_b, y_s])
        hidden = LeakyReLU(0.2) (hidden)

        hidden = Conv2D(512,(3,3),padding='same') (hidden)
        y_s, y_b = A_block(w, 512)
        noise = B_block(noise_inp, 512, 4)
        hidden = Add() ([hidden,noise])
        hidden = AdaIN() ([hidden, y_b, y_s])
        hidden = LeakyReLU(0.2) (hidden)

        hidden, rgb = G_block(hidden, w, noise_inp, 512)
        x_out.append(rgb)
        hidden, rgb = G_block(hidden, w, noise_inp, 512)
        x_out.append(rgb)
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

    def _Get_Discriminator(self):
        x = Input(shape = (self.img_height,self.img_width,self.channels))
        hidden = Conv2D(64,(1,1),padding='same') (x)
        hidden = D_block(x, 64)
        hidden = D_block(hidden, 128)
        hidden = D_block(hidden, 256)
        hidden = D_block(hidden, 512)
        hidden = D_block(hidden, 512)
        hidden = D_block(hidden, 512)
        hidden = Flatten() (hidden)
        x_out = Dense(1) (hidden)
        model = Model(x,x_out)
        return model

    def _get_model(self):
        F = self._Get_Mapping_Network()
        G = self._Get_Generator()
        D = self._Get_Discriminator()
        D.trainable = False
        z = Input(shape = (512,))
        noise_inp = Input(shape = (self.img_height,self.img_width,1))
        hidden = G([F(z),noise_inp])
        x_out = D(hidden)
        GD = Model([z,noise_inp],x_out)
        FG = Model([z,noise_inp],hidden)
        GD.summary()
        return F, G, D, FG, GD

    def __init__(self, batch_size, img_height, img_width, channels, ImageGen, train_new = False, path = None, lr = 5e-5):
        self.lamda = 10
        self.reals, _, _ = ImageGen.next_batch_Discriminator()
        self.z, self.noise = ImageGen.next_batch_Generator()
        self.fakes = None
        self.batch_size = batch_size
        self.path = path
        self.lr = lr
        self.const = None

        self.z_visual = None
        self.noise_visual = None

        self.g_loss = []
        self.d_loss = []


        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels

        if (train_new):
            self.const = np.random.normal(size = (4,4,512))
        else:
            with h5py.File(os.path.join(path, 'Model.h5'), 'r') as f:
                self.const = np.array(f["const"])


        self.const_tensor = replicate(self.const, self.batch_size)
        self.F_network, self.Generator, self.Discriminator, self.FG, self.Stacked_model = self._get_model()

        self.optimizer_D = tf.keras.optimizers.Adam(learning_rate = self.lr, beta_1 = 0, beta_2 = 0.9)
        self.optimizer_G = tf.keras.optimizers.Adam(learning_rate = self.lr, beta_1 = 0, beta_2 = 0.9)

        self.Discriminator.trainable = True
        self.train_on_batch_D()
        self.Discriminator.trainable = False
        self.train_on_batch_G()

        if not(train_new):
            self.load_model(path)

    def get_iteration(self):
        return np.array(self.optimizer_G.iterations)

    def set_Discriminator_trainable(self, flag):
        for layer in self.Discriminator.layers:
            layer.trainable = flag
        return

    def set_Generator_trainable(self, flag):
        for layer in self.FG.layers:
            layer.trainable = flag
        return

    def train_on_batch_G(self):
        with tf.GradientTape() as tape:
            logits = self.Stacked_model([self.z,self.noise], training = True)
            loss_value = WGAN_loss_G(logits)
        grads = tape.gradient(loss_value, self.FG.trainable_weights)

        self.optimizer_G.apply_gradients(zip(grads, self.FG.trainable_weights))
        return float(loss_value)

    def train_on_batch_D(self):
        with tf.GradientTape() as tape:
            self.fakes = self.FG([self.z,self.noise],training = True)
            loss_value = WGAN_loss_D(self)
        grads = tape.gradient(loss_value, self.Discriminator.trainable_weights)

        self.optimizer_D.apply_gradients(zip(grads, self.Discriminator.trainable_weights))
        return float(loss_value)
    
    def save_model(self, path):

        print("Saving...",end='')
        with h5py.File(os.path.join(path, 'Model.h5'), 'w') as f: 
            dset = f.create_dataset("const", data = self.const)
            dset = f.create_dataset("z_visual", data = self.z_visual)
            dset = f.create_dataset("noise_visual", data = self.noise_visual)
            dset = f.create_dataset('d_loss', data= self.d_loss)
            dset = f.create_dataset('g_loss', data= self.g_loss)
        self.F_network.save_weights(os.path.join(path, 'Mapping_Network.h5'))
        self.Generator.save_weights(os.path.join(path, 'Generator.h5'))
        self.Discriminator.save_weights(os.path.join(path, 'Discriminator.h5'))

        with open(os.path.join(path, "Optimizer_D.pkl"), "wb") as f:
            pickle.dump(self.optimizer_D.get_weights(), f)
        with open(os.path.join(path, "Optimizer_G.pkl"), "wb") as f:
            pickle.dump(self.optimizer_G.get_weights(), f)

        print("Done!")

    def load_model(self, path):
        with h5py.File(os.path.join(path, 'Model.h5'),'r') as f:
            self.z_visual = np.array(f["z_visual"])
            self.noise_visual = np.array(f["noise_visual"])
            self.d_loss = list(np.array(f['d_loss']))
            self.g_loss = list(np.array(f['g_loss']))

        with open(os.path.join(path, "Optimizer_D.pkl"),"rb") as f:
            Optimizer_weights = pickle.load(f)
            self.optimizer_D.set_weights(Optimizer_weights)

        with open(os.path.join(path, "Optimizer_G.pkl"), "rb") as f:
            Optimizer_weights = pickle.load(f)
            self.optimizer_G.set_weights(Optimizer_weights)
        
        self.F_network.load_weights(os.path.join(path,'Mapping_Network.h5'))
        self.Generator.load_weights(os.path.join(path, 'Generator.h5'))
        self.Discriminator.load_weights(os.path.join(path, 'Discriminator.h5'))
