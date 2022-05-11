import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, AveragePooling2D, UpSampling2D, Dense, Flatten, Add, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from models.losses import consistency_loss, GAN_loss, identity_loss

def G_step(G, GG, D, optimizer, x, x_id):
    lambda_cycle = 10
    lambda_identity = 0.4
    with tf.GradientTape() as tape:
        generated = G(x, training = True)
        y_id = G(x_id, training = True)
        logits = D(generated, training = True)
        y_true = np.ones(shape = (x.shape[0],1))
        loss = GAN_loss(y_true, logits) + lambda_cycle * consistency_loss(GG, x, generated) + lambda_identity * identity_loss(x_id, y_id)
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
    out = AveragePooling2D() (hidden)
    return out

def US_ConvBlock(num_filter, x):
    hidden = UpSampling2D(interpolation = 'bilinear') (x)
    out = Conv2D(num_filter, (3,3), padding = 'same', activation = LeakyReLU(0.2)) (hidden)
    return out

def Res_ConvBlock(num_filter, x):
    hidden = Conv2D(num_filter, (3,3), padding = 'same', activation = LeakyReLU(0.2)) (x)
    hidden = Conv2D(num_filter, (3,3), padding = 'same', activation = LeakyReLU(0.2)) (hidden)
    out = Add() ([x, hidden])
    return out

def get_discriminator(img_shape):
    num_filters = [64, 128, 256, 512]
    x = Input(shape = img_shape)

    hidden = Conv2D(num_filters[0], (1,1), padding = 'same', activation = LeakyReLU(0.2)) (x)

    for idx in range(len(num_filters)):
        hidden = DS_ConvBlock(num_filters[idx], hidden)

    hidden = Flatten() (hidden)
    out = Dense(1, activation = 'sigmoid') (hidden)
    return Model(x, out)

def get_unetgenerator(img_shape):
    pass

def get_generator(img_shape):
    num_filters = [64, 128, 256]

    x = Input(shape = img_shape)
    hidden = Conv2D(64, (1,1), padding = 'same', activation = LeakyReLU(0.2)) (x)
    hidden = DS_ConvBlock(128, hidden)
    hidden = DS_ConvBlock(256, hidden)

    for idx in range(9):
        hidden = Res_ConvBlock(256, hidden)

    hidden = US_ConvBlock(128, hidden)
    hidden = US_ConvBlock(64, hidden)

    out = Conv2DTranspose(3, (1,1), padding = 'same', activation = 'tanh') (hidden)

    return Model(x, out)

def get_optimizer(lr_base = 0.0001):
    optimizer = Adam(lr_base, beta_1 = 0.5)
    return optimizer

def save_checkpoint(model, path):
    model.save_weights(path)
    return model

def load_checkpoint(model, path):
    model.load_weights(path)
    return model