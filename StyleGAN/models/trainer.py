import os
import pickle
import cv2
import h5py
import numpy as np
import tensorflow as tf
from models.losses import WGAN_Generator_loss, WGAN_Disciminator_loss   

def save_checkpoint(model, path):
    model.save_weights(path)

class Trainer():
    def __init__(self, lr, loss = 0):
        self.z_visual = None
        self.noise_visual = None
        self.BatchGen = None
        self.d_loss = []
        self.g_loss = []
        self.FG = None
        self.D = None
        self.optimizer_D = tf.keras.optimizers.Adam(lr=lr, beta_1=0.0, beta_2=0.9)
        self.optimizer_G = tf.keras.optimizers.Adam(lr=lr, beta_1=0.0, beta_2=0.9)
        self.loss = loss

    def get_preview(self, FG, num_images, random = True):
        if (random):
            zs, noises = self.BatchGen.next_batch_Generator(batch_size = num_images)
        else:
            zs, noises = self.z_visual, self.noise_visual

        preview_imgs = []
        for idx in range(num_images):
            z = np.reshape(zs[idx], (1, self.BatchGen.latent_size))
            noise = np.reshape(noises[idx], (1, self.BatchGen.img_size[0], self.BatchGen.img_size[1], 1))
            preview_imgs.append(cv2.cvt(FG.predict([z,noise])[0], cv2.COLOR_RGB2BGR))
        return preview_imgs

    def get_num_iteration(self):
        return np.int(self.optimizer_G.iterations)

    def set_BatchGen(self, BG):
        self.BatchGen = BG

    def init_visualization(self, num_images):
        self.z_visual, self.noise_visual = self.BatchGen.next_batch_Generator(batch_size = num_images)

    def generator_step(self, x):
        with tf.GradientTape() as tape:
            logits = self.D(self.FG(x, training = True), training = True)
            loss_value = WGAN_Generator_loss(logits)
            
            grads = tape.gradient(loss_value, self.FG.trainable_weights)

        self.optimizer_G.apply_gradients(zip(grads, self.FG.trainable_weights))
        return float(loss_value)

    def discriminator_step(self, x):
        with tf.GradientTape() as tape:
            fakes = self.FG([x[1],x[2]],training = True)
            loss_value = WGAN_Disciminator_loss(self.D, x[0], fakes)

            grads = tape.gradient(loss_value, self.D.trainable_weights)

        self.optimizer_D.apply_gradients(zip(grads, self.D.trainable_weights))
        return float(loss_value)
        

    def load(self, path):
        with h5py.File(os.path.join(path,"trainer_checkpoint.h5"), "r") as f:
            self.z_visual = np.array(f["z_visual"])
            self.noise_visual = np.array(f["noise_visual"])
            self.d_loss = list(np.array(f['d_loss']))
            self.g_loss = list(np.array(f['g_loss']))

    def load_optimizers(self, path, FG, D):
        z, noise = self.BatchGen.next_batch_Generator()
        self.generator_step([z, noise])

        reals, z, noise = self.BatchGen.next_batch_Discriminator()
        self.discriminator_step([reals, z, noise])
        
        with open(os.path.join(path, "Optimizer_D.pkl"),"rb") as f:
            Optimizer_weights = pickle.load(f)
            self.optimizer_D.set_weights(Optimizer_weights)

        with open(os.path.join(path, "Optimizer_G.pkl"), "rb") as f:
            Optimizer_weights = pickle.load(f)
            self.optimizer_G.set_weights(Optimizer_weights)

    def save(self, path):
        with h5py.File(os.path.join(path, 'trainer_checkpoint.h5'), 'w') as f: 
            dset = f.create_dataset("z_visual", data=self.z_visual)
            dset = f.create_dataset("noise_visual", data=self.noise_visual)
            dset = f.create_dataset('d_loss', data=self.d_loss)
            dset = f.create_dataset('g_loss', data=self.g_loss)

        with open(os.path.join(path, "Optimizer_D.pkl"), "wb") as f:
            pickle.dump(self.optimizer_D.get_weights(), f)
        with open(os.path.join(path, "Optimizer_G.pkl"), "wb") as f:
            pickle.dump(self.optimizer_G.get_weights(), f)
     