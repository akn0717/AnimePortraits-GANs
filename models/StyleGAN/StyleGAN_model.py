from tensorflow.python.keras.layers.convolutional import Cropping2D, UpSampling2D
from loss_function import *
from keras.models import Model
from keras.layers import Conv2D, Dense, Activation, Flatten, Reshape, Input, UpSampling2D, Add
from keras.layers import LeakyReLU, Cropping2D, AveragePooling2D
from keras.optimizers import Adam
from AdaIN import AdaInstanceNormalization

import h5py

import os

def G_block(x, y_s, y_b, noise, filter):
    hidden = UpSampling2D() (x)
    hidden = Conv2D(filter,(3,3),padding = 'same') (hidden)
    hidden = Activation('relu') (hidden)
    hidden = Conv2D(filter,(3,3),padding = 'same') (hidden)
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

def replicate(const, batch_size):
    ans = []
    for _ in range(batch_size):
        ans.append(const)
    ans = tf.constant(np.array(ans))
    return ans

class StyleGAN():
    def _Get_Generator(self):
        x_out = []
        z = Input(shape = (512,))
        FC = Dense(512, activation = 'relu') (z)
        FC = Dense(512, activation = 'relu') (FC)
        FC = Dense(512, activation = 'relu') (FC)
        FC = Dense(512, activation = 'relu') (FC)
        FC = Dense(512, activation = 'relu') (FC)
        FC = Dense(512, activation = 'relu') (FC)
        FC = Dense(512, activation = 'relu') (FC)
        w = Dense(512, activation = 'relu') (FC)

        noise_inp = Input(shape = (self.img_height,self.img_width,1))

        x = self.const_tensor
        y_s, y_b = A_block(w, 512)
        noise = B_block(noise_inp, 512, 4)
        hidden = Add() ([x,noise])
        hidden = AdaInstanceNormalization() ([hidden, y_b, y_s])
        hidden = Activation('relu') (hidden)
        y_s, y_b = A_block(w, 256)
        noise = B_block(noise_inp, 256, 8)
        hidden, rgb = G_block(hidden, y_s, y_b, noise, 256)
        x_out.append(rgb)
        y_s, y_b = A_block(w, 128)
        noise = B_block(noise_inp, 128, 16)
        hidden, rgb = G_block(hidden, y_s, y_b, noise, 128)
        x_out.append(rgb)
        y_s, y_b = A_block(w, 64)
        noise = B_block(noise_inp, 64, 32)
        hidden, rgb = G_block(hidden, y_s, y_b, noise, 64)
        x_out.append(rgb)
        y_s, y_b = A_block(w, 32)
        noise = B_block(noise_inp, 32, 64)
        hidden, rgb = G_block(hidden, y_s, y_b, noise, 32)
        x_out.append(rgb)
        y_s, y_b = A_block(w, 16)
        noise = B_block(noise_inp, 16, 128)
        hidden, rgb = G_block(hidden, y_s, y_b, noise, 16)
        x_out.append(rgb)
        y_s, y_b = A_block(w, 8)
        noise = B_block(noise_inp, 8, 256)
        hidden, rgb = G_block(hidden, y_s, y_b, noise, 8)
        x_out.append(rgb)
        x_out = Add() (x_out)
        x_out = Activation('tanh') (x_out)
        model = Model([z,noise_inp],x_out)
        return model

    def _Get_Discriminator(self):
        x = Input(shape = (self.img_height,self.img_width,self.channels))
        hidden = Conv2D(16,(1,1),padding='same') (x)
        hidden = D_block(x, 16)
        hidden = D_block(hidden, 32)
        hidden = D_block(hidden, 64)
        hidden = D_block(hidden, 128)
        hidden = D_block(hidden, 256)
        hidden = D_block(hidden, 512)
        hidden = Flatten() (hidden)
        x_out = Dense(1) (hidden)
        model = Model(x,x_out)
        return model

    def _get_model(self):
        G = self._Get_Generator()
        D = self._Get_Discriminator()
        D.trainable = False
        z = Input(shape = (512,))
        noise_inp = Input(shape = (self.img_height,self.img_width,1))
        hidden = G([z,noise_inp])
        x_out = D(hidden)
        GD = Model([z,noise_inp],x_out)
        GD.summary()
        return G, D, GD

    def __init__(self, batch_size, img_height, img_width, channels, path):
        self.lamda = 10
        self.reals = None
        self.z = None
        self.noise = None
        self.fakes = None
        self.batch_size = batch_size
        self.epoch = 0
        self.path = path

        self.const = None
        self.load_const(path)

        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels

        

        self.const_tensor = replicate(self.const, self.batch_size)
        self.Generator, self.Discriminator, self.Stacked_model = self._get_model()
        
        self.optimizer_D = Adam(learning_rate = 0.0001, beta_1 = 0, beta_2 = 0.9)
        self.optimizer_G = Adam(learning_rate = 0.0001, beta_1 = 0, beta_2 = 0.9)

    def train_on_batch_G(self):
        with tf.GradientTape() as tape:
            logits = self.Stacked_model([self.z,self.noise], training = True)
            loss_value = WGAN_loss_G(logits)
        grads = tape.gradient(loss_value, self.Generator.trainable_weights)

        self.optimizer_G.apply_gradients(zip(grads, self.Generator.trainable_weights))
        return float(loss_value)

    def train_on_batch_D(self):
        with tf.GradientTape() as tape:
            self.fakes = self.Generator([self.z,self.noise],training = True)
            loss_value = WGAN_loss_D(self)
        grads = tape.gradient(loss_value, self.Discriminator.trainable_weights)

        self.optimizer_D.apply_gradients(zip(grads, self.Discriminator.trainable_weights))
        return float(loss_value)
    
    def save_model(self, path):
        array = np.array([self.epoch, self.img_height, self.img_width, self.batch_size],dtype = int)

        with h5py.File(os.path.join(path, 'Model.h5'), 'w') as f: 
            dset = f.create_dataset("model_details", data = array)
        self.Generator.save_weights(os.path.join(path, 'Generator.h5'))
        self.Discriminator.save_weights(os.path.join(path, 'Discriminator.h5'))

    def load_model(self, path):
        with h5py.File(os.path.join(path, 'Model.h5'),'r') as f:
            data = f['model_details']
            self.epoch = data[0]
            self.img_height, self.img_width = data[1], data[2]
            self.batch_size = data[3]
        self.Generator.load_weights(os.path.join(path, 'Generator.h5'))
        self.Discriminator.load_weights(os.path.join(path, 'Discriminator.h5'))

    def save_const(self, path):
        const = np.reshape(self.const,(4*4*512))
        np.savetxt(os.path.join(path, 'key_const.bin'),const,delimiter=',')

    def load_const(self, path):
        if os.path.exists(os.path.join(path, 'key_const.bin')):
            const = np.loadtxt(os.path.join(path, 'key_const.bin'),delimiter=',')
            const = np.reshape(const,(4,4,512))
        else:
            const = np.random.normal(size = (4,4,512))

        self.const = const

        if not(os.path.exists(os.path.join(path, 'key_const.bin'))):
            self.save_const(path)