import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, AveragePooling2D, UpSampling2D, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from models.losses import consistency_loss, GAN_loss

def G_step(G, GG, D, optimizer, x):
    with tf.GradientTape() as tape:
        generated = G(x, training = True)
        logits = D(generated, training = True)
        y_true = np.ones(shape = (x.shape[0],1))
        loss = GAN_loss(y_true, logits) + 10 * consistency_loss(GG, x, generated)
        grads = tape.gradient(loss, G.trainable_weights)
    optimizer.apply_gradients(zip(grads, G.trainable_weights))
    return float(loss)

def D_step(D, optimizer, x, y_true):
    with tf.GradientTape() as tape:
        logits = D(x, training = True)
        loss = GAN_loss(y_true, logits)
        grads = tape.gradient(loss, D.trainable_weights)
    optimizer.apply_gradients(zip(grads, D.trainable_weights))
    return float(loss)

def DS_ConvBlock(num_filter, x):
    hidden = Conv2D(num_filter, (3,3), padding = 'same', activation = LeakyReLU(0.2)) (x)
    hidden = Conv2D(num_filter, (3,3), padding = 'same', activation = LeakyReLU(0.2)) (hidden)
    out = AveragePooling2D() (hidden)
    return out

def US_ConvBlock(num_filter, x):
    hidden = UpSampling2D(interpolation = 'bilinear') (x)
    hidden = Conv2D(num_filter, (3,3), padding = 'same', activation = LeakyReLU(0.2)) (hidden)
    out = Conv2D(num_filter, (3,3), padding = 'same', activation = LeakyReLU(0.2)) (hidden)
    return out

def get_discriminator(H = 64, W = 64, C = 3):
    num_filters = [64, 128, 256, 512]
    x = Input(shape = (H, W, C))

    hidden = Conv2D(num_filters[0], (1,1), padding = 'same', activation = LeakyReLU(0.2)) (x)

    for idx in range(len(num_filters)):
        hidden = DS_ConvBlock(num_filters[idx], hidden)

    hidden = Flatten() (hidden)
    out = Dense(1, activation = 'sigmoid') (hidden)
    return Model(x, out)

def get_unetgenerator(H, W, C):
    pass

def get_generator(H = 64, W = 64, C = 3):
    num_filters = [64, 128, 256, 512]
    x = Input((H, W, C))
    hidden = Conv2D(num_filters[0], (1,1), padding = 'same') (x)

    for idx in range(len(num_filters)):
        hidden = DS_ConvBlock(num_filters[idx], hidden)
    
    for idx in range(len(num_filters) - 1, -1, -1):
        hidden = US_ConvBlock(num_filters[idx], hidden)
    

    out = Conv2D(3, (1,1), padding = 'same', activation = 'tanh') (hidden)
    
    return Model(x, out)

def get_optimizer(path = None, model = None, img_shape = (64, 64, 3), lr_base = 0.0001):
    optimizer = Adam(lr_base, beta_1 = 0.5)
    return optimizer
        
