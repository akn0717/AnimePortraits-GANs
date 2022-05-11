import io
import os
import random
import h5py
import cv2
import numpy as np
from PIL import Image

from utils.imglib import load_sampling, standardize_image, rand_crop

def load_bfile(h5_file, name, img_shape, resize = False, random_crop = True):
    img = np.asarray(Image.open(io.BytesIO(np.array(h5_file[name]))))
    if (img.shape[0]!=img_shape[0] or img.shape[1] != img_shape[1]):
        if (resize):
            img = cv2.resize(img, (img_shape[0], img_shape[1]))
        if (random_crop):
            img = rand_crop(img, img_shape)
    return img

def load_bfiles(h5_file, names, img_shape, resize = False, random_crop = True):
    img = []
    for name in names:
        img.append(load_bfile(h5_file, name, img_shape, resize = False, random_crop = True))
    return img

class BatchGen():
    def __init__(self, path_A, path_B, img_size, resize = False, random_crop = True):

        self.resize = resize
        self.random_crop = random_crop

        self.path_A = path_A
        self.path_B = path_B

        self.A_file = None
        self.B_file = None
        
        if os.path.isdir(path_A):
            self.A_keys = [f for f in os.listdir(self.path_A) if os.path.isfile(os.path.join(self.path_A, f))]
        else:
            self.A_file = h5py.File(path_A, 'r')
            self.A_keys = self.A_file.keys()

        if os.path.isdir(path_B):
            self.B_keys = [f for f in os.listdir(self.path_B) if os.path.isfile(os.path.join(self.path_B, f))]
        else:
            self.B_file = h5py.File(path_B, 'r')
            self.B_keys = self.B_file.keys()
        
    def get_A_samples(self, batch_size, img_shape):
        A_key_samples = random.sample(self.A_keys, batch_size)

        if (self.A_file != None):
            A_samples = load_bfiles(self.A_file, A_key_samples, img_shape, self.resize, self.random_crop)
        else:
            A_samples = np.array(load_sampling(self.path_A, A_key_samples, H = img_shape[0], W = img_shape[1]))

        A_samples = standardize_image(A_samples)

        return A_samples
    
    def get_B_samples(self, batch_size, img_shape):
        B_key_samples = random.sample(self.B_keys, batch_size)

        if (self.B_file != None):
            B_samples = load_bfiles(self.B_file, B_key_samples, img_shape, self.resize, self.random_crop)
        else:
            B_samples = np.array(load_sampling(self.path_B, B_key_samples, H = img_shape[0], W = img_shape[1]))

        B_samples = standardize_image(B_samples)
        
        return B_samples

    def get_size(self):
        return len(self.A_keys), len(self.B_keys)