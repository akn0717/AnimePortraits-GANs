from models.loss_function import *
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Activation, Flatten, Reshape, Conv2DTranspose, Input
from keras.layers import LeakyReLU
from keras_contrib.layers import InstanceNormalization
from keras.optimizers import Adam

class WGANGP_model():
    def _Get_Generator(self):
        x = Input(shape = (100,))
        hidden = Dense(8192) (x)
        hidden = Reshape((4,4,512)) (hidden)
        hidden = InstanceNormalization() (hidden)
        hidden = Activation('relu') (hidden)
        hidden = Conv2DTranspose(256,(5,5),strides = (2,2),padding = 'same') (hidden)
        hidden = InstanceNormalization() (hidden)
        hidden = Activation('relu') (hidden)
        hidden = Conv2DTranspose(128,(5,5),strides = (2,2),padding = 'same') (hidden)
        hidden = InstanceNormalization() (hidden)
        hidden = Activation('relu') (hidden)
        hidden = Conv2DTranspose(64,(5,5),strides = (2,2),padding = 'same') (hidden)
        hidden = InstanceNormalization() (hidden)
        hidden = Activation('relu') (hidden)
        x_out = Conv2DTranspose(3,(5,5),strides = (2,2),padding = 'same', activation = 'tanh') (hidden)
        model = Model(x,x_out)
        return model

    def _Get_Discriminator(self):
        x = Input(shape = (64,64,3))
        hidden = Conv2D(64,(5,5),strides = (2,2),padding = 'same') (x)
        hidden = InstanceNormalization() (hidden)
        hidden = LeakyReLU(0.2) (hidden)
        hidden = Conv2D(128,(5,5),strides = (2,2),padding = 'same') (hidden)
        hidden = InstanceNormalization() (hidden)
        hidden = LeakyReLU(0.2) (hidden)
        hidden = Conv2D(256,(5,5),strides = (2,2),padding = 'same') (hidden)
        hidden = InstanceNormalization() (hidden)
        hidden = LeakyReLU(0.2) (hidden)
        hidden = Conv2D(512,(5,5),strides = (2,2),padding = 'same') (hidden)
        hidden = InstanceNormalization() (hidden)
        hidden = LeakyReLU(0.2) (hidden)
        hidden = Flatten() (hidden)
        x_out = Dense(1) (hidden)
        model = Model(x,x_out)
        return model

    def _get_model(self):
        G = self._Get_Generator()
        D = self._Get_Discriminator()
        D.trainable = False
        GD = Sequential()
        GD.add(G)
        GD.add(D)
        return G, D, GD

    def __init__(self):
        self.Generator, self.Discriminator, self.Stacked_model = self._get_model()
        self.lamda = 10
        self.reals = None
        self.noise = None
        self.fakes = None
        self.optimizer_D = Adam(learning_rate = 0.0001, beta_1 = 0, beta_2 = 0.9)
        self.optimizer_G = Adam(learning_rate = 0.0001, beta_1 = 0, beta_2 = 0.9)

    def train_on_batch_G(self):
        with tf.GradientTape() as tape:
            logits = self.Stacked_model(self.noise, training = True)
            loss_value = WGAN_loss_G(logits)
        grads = tape.gradient(loss_value, self.Generator.trainable_weights)

        self.optimizer_G.apply_gradients(zip(grads, self.Generator.trainable_weights))
        return float(loss_value)

    def train_on_batch_D(self):
        with tf.GradientTape() as tape:
            self.fakes = self.Generator(self.noise,training = True)
            loss_value = WGAN_loss_D(self)
        grads = tape.gradient(loss_value, self.Discriminator.trainable_weights)

        self.optimizer_D.apply_gradients(zip(grads, self.Discriminator.trainable_weights))
        return float(loss_value)