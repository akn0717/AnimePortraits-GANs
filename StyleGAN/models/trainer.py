import json
import os
import pickle
import cv2
import h5py
import numpy as np
from sklearn.metrics import top_k_accuracy_score
import tensorflow as tf
from models.generator import predict_batch
from models.losses import LSGAN_loss, WGAN_Generator_loss, WGAN_Disciminator_loss
from config import *
from utils.imglib import destandardize_image

def save_checkpoint(model, path):
    model.save_weights(path)


class Trainer():
    def __init__(self, lr, loss_fn = 0):
        self.z_visual = None
        self.noise_visual = None
        self.BatchGen = None
        self.d_loss = []
        self.g_loss = []
        self.FG = None
        self.F = None
        self.D = None
        self.optimizer_D = tf.keras.optimizers.Adam(lr=lr, beta_1=0.0, beta_2=0.9)
        self.optimizer_G = tf.keras.optimizers.Adam(lr=lr, beta_1=0.0, beta_2=0.9)
        self.loss_fn = loss_fn
        self.iterations = 0

    def get_preview(self, num_images, random = True):
        if (random):
            z, noise = self.BatchGen.next_batch_Generator(num_images)
        else:
            z, noise = self.z_visual, self.noise_visual

        samples = destandardize_image(predict_batch(self.FG, [z, noise]))
        for idx in range(len(samples)):
            samples[idx] = cv2.cvtColor(samples[idx], cv2.COLOR_RGB2BGR)
        return samples

    def get_num_iteration(self):
        return self.iterations

    def set_BatchGen(self, BG):
        self.BatchGen = BG

    def init_visualization(self, num_images):
        self.z_visual, self.noise_visual = self.BatchGen.next_batch_Generator(num_images)

    def generator_step(self, x):
        
        grads = None
        loss = 0
        z, noise = x[0], x[1]
        for idx in range(0, BATCH_SIZE, MINI_BATCH_SIZE):
            z_minibatch = z[idx:min(idx+MINI_BATCH_SIZE, BATCH_SIZE)]
            noise_minibatch = noise[idx:min(idx+MINI_BATCH_SIZE, BATCH_SIZE)]

            logits = self.D(self.FG([z_minibatch, noise_minibatch], training = True), training = True)

            res_topk = tf.math.top_k(tf.reshape(logits, shape = (MINI_BATCH_SIZE,)), TOP_K)
            z_minibatch = z_minibatch[res_topk.indices.numpy()]
            noise_minibatch = noise_minibatch[res_topk.indices.numpy()]

            with tf.GradientTape() as tape:
                
                generates = self.FG([z_minibatch, noise_minibatch], training = True)
                logits = self.D(generates, training = True)
                
                if (self.loss_fn==0):
                    loss_minibatch = WGAN_Generator_loss(logits)
                else:
                    loss_minibatch = LSGAN_loss(tf.ones(shape = logits.shape), logits)

                mini_batchgrads = tape.gradient(loss_minibatch, self.FG.trainable_weights)

            
            if grads is None:
                grads = mini_batchgrads
            else:
                for idx in range(len(grads)):
                    grads[idx] = tf.add(grads[idx], mini_batchgrads[idx])

            loss += loss_minibatch

        for idx in range(len(grads)): 
            grads[idx] = tf.divide(grads[idx], tf.math.ceil(1.*BATCH_SIZE / MINI_BATCH_SIZE)+1e-8)

        self.optimizer_G.apply_gradients(zip(grads, self.FG.trainable_weights))
        self.iterations += 1
        return float(loss/ (1.*BATCH_SIZE / MINI_BATCH_SIZE))

    def discriminator_step(self, x):
        grads = None
        loss = 0
        reals, z, noise = x[0], x[1], x[2]
        for idx in range(0, BATCH_SIZE, MINI_BATCH_SIZE):
            with tf.GradientTape() as tape:
                reals_minibatch = reals[idx:min(idx+MINI_BATCH_SIZE, BATCH_SIZE)]
                z_minibatch = z[idx:min(idx+MINI_BATCH_SIZE, BATCH_SIZE)]
                noise_minibatch = noise[idx:min(idx+MINI_BATCH_SIZE, BATCH_SIZE)]
                fakes_minibatch = self.FG([z_minibatch, noise_minibatch],training = True)

                if (self.loss_fn == 0):
                    loss_minibatch = WGAN_Disciminator_loss(self.D, reals_minibatch, fakes_minibatch)
                else:
                    x_pred = tf.concat([fakes_minibatch, reals_minibatch], axis = 0)
                    y_pred = self.D(x_pred)
                    y_true = tf.concat([tf.zeros(shape = (fakes_minibatch.shape[0],1)), tf.ones(shape = (reals_minibatch.shape[0],1))], axis = 0)
                    loss_minibatch = LSGAN_loss(y_pred, y_true)

                mini_batchgrads = tape.gradient(loss_minibatch, self.D.trainable_weights)

            if grads is None:
                grads = mini_batchgrads
            else:
                for idx in range(len(grads)):
                    grads[idx] = tf.add(grads[idx], mini_batchgrads[idx])

            loss += loss_minibatch 
        for idx in range(len(grads)): 
            grads[idx] = tf.divide(grads[idx], tf.math.ceil(1.*BATCH_SIZE / (MINI_BATCH_SIZE+1e-8)))

        self.optimizer_D.apply_gradients(zip(grads, self.D.trainable_weights))
        return float(loss/ (1.*BATCH_SIZE / MINI_BATCH_SIZE))
        
    def load_optimizers(self, path):
        if os.path.isfile(os.path.join(path, 'optimizer_D.pkl')):
            zeros_grads = [tf.zeros_like(w) for w in self.D.trainable_weights]
            self.optimizer_D.apply_gradients(zip(zeros_grads, self.D.trainable_weights))
            with open(os.path.join(path, 'optimizer_D.pkl'), 'rb') as f:
                opt_weights = pickle.load(f)
            self.optimizer_D.set_weights(opt_weights)

        if os.path.isfile(os.path.join(path, 'optimizer_G.pkl')):
            zeros_grads = [tf.zeros_like(w) for w in self.FG.trainable_weights]
            self.optimizer_G.apply_gradients(zip(zeros_grads, self.FG.trainable_weights))
            with open(os.path.join(path, 'optimizer_G.pkl'), 'rb') as f:
                opt_weights = pickle.load(f)
            self.optimizer_G.set_weights(opt_weights)

    def load(self, path):

        self.load_optimizers(path)
        with h5py.File(os.path.join(path,"trainer_checkpoint.h5"), "r") as f:
            self.F.load_weights(os.path.join(path,'mapping_weights.h5'))
            self.G.load_weights(os.path.join(path,'generator_weights.h5'))
            self.D.load_weights(os.path.join(path,'discriminator_weights.h5'))
            self.z_visual = np.array(f["z_visual"])
            self.noise_visual = np.array(f["noise_visual"])
            self.d_loss = list(np.array(f['d_loss']))
            self.g_loss = list(np.array(f['g_loss']))

        with open(os.path.join(path,"log.json"), "r") as f:
            data = json.load(f)
            self.iterations = data['iterations']


    def save(self, path, configs):
        os.makedirs(path, exist_ok = True)
        with h5py.File(os.path.join(path, 'trainer_checkpoint.h5'), 'w') as f: 
            dset = f.create_dataset("z_visual", data=self.z_visual)
            dset = f.create_dataset("noise_visual", data=self.noise_visual)
            dset = f.create_dataset('d_loss', data=self.d_loss)
            dset = f.create_dataset('g_loss', data=self.g_loss)

        configs['iterations'] = self.iterations
        with open(os.path.join(path,"log.json"), "w") as f:
                json.dump(configs, f)

        with open(os.path.join(path, 'optimizer_D.pkl'), 'wb') as f:
            pickle.dump(self.optimizer_D.get_weights(), f)

        with open(os.path.join(path, 'optimizer_G.pkl'), 'wb') as f:
            pickle.dump(self.optimizer_G.get_weights(), f)

        configs['iterations'] = self.iterations
        save_checkpoint(self.F, os.path.join(path, "mapping_weights.h5"))
        save_checkpoint(self.G, os.path.join(path, "generator_weights.h5"))
        save_checkpoint(self.D, os.path.join(path, "discriminator_weights.h5"))
        
     