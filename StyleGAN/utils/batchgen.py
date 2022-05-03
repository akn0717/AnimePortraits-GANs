import io
import os
import random
import h5py
import cv2
import numpy as np
from PIL import Image

from utils.imglib import load_sampling, standardize_image

def load_bfile(h5_file, name):
    img = np.asarray(Image.open(io.BytesIO(np.array(h5_file[name]))))
    return img

def load_bfiles(h5_file, names):
    img = []
    for name in names:
        img.append(load_bfile(h5_file, name))
    return img

class BatchGen():
    def __init__(self, path, img_size, batch_size, latent_size):

        self.path = path
        self.h5_file = None
        
        if os.path.isdir(path):
            self.keys = [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]
        else:
            self.h5_file = h5py.File(path, 'r')
            self.keys = self.h5_file.keys()
        
        self.img_size = img_size
        self.batch_size = batch_size
        self.latent_size = latent_size
        return
    
    def next_batch_Generator(self, batch_size = None, img_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        
        if img_size is None:
            img_size = self.img_size

        z = np.random.normal(size = (batch_size, self.latent_size))
        noise = np.random.normal(size = (batch_size, img_size[0], img_size[1],1))
        return z, noise

    def next_batch_Discriminator(self, batch_size = None, img_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        
        if img_size is None:
            img_size = self.img_size

        real_samples = random.sample(self.keys, batch_size)
        if (self.h5_file != None):
            real_samples = standardize_image(load_bfiles(self.h5_file, real_samples))
        else:
            real_samples = np.array(load_sampling(self.path, real_samples, H = img_size[0], W = img_size[1]))
            
        z = np.random.normal(size = (batch_size, self.latent_size))
        noise = np.random.normal(size = (batch_size, img_size[0], img_size[1],1))

        return real_samples, z, noise

    def get_size(self):
        return len(self.keys)