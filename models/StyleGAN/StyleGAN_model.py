from tensorflow.python.keras.layers.convolutional import Cropping2D, UpSampling2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
from loss_function import *
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Activation, Flatten, Reshape, Input, UpSampling2D, Add
from keras.layers import LeakyReLU, Cropping2D, AveragePooling2D
from keras.optimizers import Adam
from AdaIN import AdaInstanceNormalization

def g_block(x, y_s, y_b, noise, filter):
    hidden = UpSampling2D() (x)
    hidden = Conv2D(filter,(3,3),padding = 'same') (hidden)
    hidden = Activation('relu') (hidden)
    hidden = Conv2D(filter,(3,3),padding = 'same') (hidden)
    hidden = Add() ([hidden,noise])
    hidden = AdaInstanceNormalization() ([hidden, y_b, y_s])
    x_out = Activation('relu') (hidden)

    to_rgb = Conv2D(3, (1,1), padding = 'same') (x_out)
    to_rgb = UpSampling2D(size = (int(64/hidden.shape[1]),int(64/hidden.shape[1]))) (to_rgb)
    return x_out, to_rgb
    
def d_block(x, filter):
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
    size = ((0,64 - size),(0,64 - size))
    out = Cropping2D(cropping = size) (noise)
    out = Conv2D(filter, (1,1), padding = 'same') (out)
    return out

def replicate(const, batch_size):
    ans = []
    for _ in range(batch_size):
        ans.append(const)
    ans = tf.constant(np.array(ans))
    return ans

class WGANGP_model():
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

        noise_inp = Input(shape = (64,64,1))

        x = self.const_tensor
        y_s, y_b = A_block(w, 512)
        noise = B_block(noise_inp, 512, 4)
        hidden = Add() ([x,noise])
        hidden = AdaInstanceNormalization() ([hidden, y_b, y_s])
        hidden = Activation('relu') (hidden)
        y_s, y_b = A_block(w, 256)
        noise = B_block(noise_inp, 256, 8)
        hidden, rgb = g_block(hidden, y_s, y_b, noise, 256)
        x_out.append(rgb)
        y_s, y_b = A_block(w, 128)
        noise = B_block(noise_inp, 128, 16)
        hidden, rgb = g_block(hidden, y_s, y_b, noise, 128)
        x_out.append(rgb)
        y_s, y_b = A_block(w, 64)
        noise = B_block(noise_inp, 64, 32)
        hidden, rgb = g_block(hidden, y_s, y_b, noise, 64)
        x_out.append(rgb)
        y_s, y_b = A_block(w, 32)
        noise = B_block(noise_inp, 32, 64)
        hidden, rgb = g_block(hidden, y_s, y_b, noise, 32)
        x_out.append(rgb)
        x_out = Add() (x_out)
        x_out = Activation('tanh') (x_out)
        model = Model([z,noise_inp],x_out)
        return model

    def _Get_Discriminator(self):
        x = Input(shape = (64,64,3))
        hidden = Conv2D(64,(1,1),padding='same') (x)
        hidden = d_block(x, 64)
        hidden = d_block(hidden, 128)
        hidden = d_block(hidden, 256)
        hidden = d_block(hidden, 512)
        hidden = Flatten() (hidden)
        x_out = Dense(1) (hidden)
        model = Model(x,x_out)
        return model

    def _get_model(self):
        G = self._Get_Generator()
        D = self._Get_Discriminator()
        D.trainable = False
        z = Input(shape = (512,))
        noise_inp = Input(shape = (64,64,1))
        hidden = G([z,noise_inp])
        x_out = D(hidden)
        GD = Model([z,noise_inp],x_out)
        GD.summary()
        return G, D, GD

    def __init__(self, batch_size, load_model = False, load_const = False):
        self.lamda = 10
        self.reals = None
        self.z = None
        self.noise = None
        self.fakes = None
        self.batch_size = batch_size

        if load_const == True:
            const = np.loadtxt('const.txt',delimiter=',')
            const = np.reshape(const,(4,4,512))
        else:
            const = np.random.normal(size = (4,4,512))

        self.const = const

        if load_const == False:
            self.save_const()

        self.const_tensor = replicate(const, self.batch_size)
        self.Generator, self.Discriminator, self.Stacked_model = self._get_model()

        if load_model == True:
            
            self.Generator.load_weights('Generator.h5')
            self.Discriminator.load_weights('Discriminator.h5')
        
        
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
    
    def save_model(self):
        self.Generator.save_weights('/content/drive/MyDrive/Generator_R.h5')
        self.Discriminator.save_weights('/content/drive/MyDrive/Discriminator_R.h5')

    def save_const(self):
        const = np.reshape(self.const,(4*4*512))
        np.savetxt('const.txt',const,delimiter=',')