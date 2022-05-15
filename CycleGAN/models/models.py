import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, AveragePooling2D, UpSampling2D, Dense, Flatten, Add, Conv2DTranspose, InputLayer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from models.losses import consistency_loss, GAN_loss, identity_loss

def G_step(G, GG, D, optimizer, x, x_id):
    lambda_cycle = 10
    lambda_identity = 0.4
    with tf.GradientTape() as tape:
        generates = G(x, training = True)
        logits = D(generates, training = True)
        y_true = np.ones(shape = logits.shape)
        loss = GAN_loss(y_true, logits) + lambda_cycle * consistency_loss(GG, x, generates) + lambda_identity * identity_loss(G, x_id)
        grads = tape.gradient(loss, G.trainable_weights)
    optimizer.apply_gradients(zip(grads, G.trainable_weights))
    return float(loss)

def D_step(D, optimizer, reals, generates):
    with tf.GradientTape() as tape:
        x = tf.concat([reals, generates], axis = 0)
        logits = D(x, training = True)
        y_true = tf.concat([tf.ones(shape = reals.shape[0] + logits.shape[1:]), tf.zeros(shape = generates.shape[0] + logits.shape[1:])], axis = 0)
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

    x = Input(shape = [None, None, 3])
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