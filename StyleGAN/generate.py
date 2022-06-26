import argparse
import os
import cv2

import numpy as np

from models.generator import get_generator
from utils.imglib import destandardize_image
from utils.plotlib import display_img

def get_w_mean(F):
    w_mean = np.zeros((1, 512))
    for _ in range(1000):
        w_mean += F(np.random.normal(size = (1, 512)))

    return w_mean / 1000

def truncate(w, w_mean, psi):
    return 1.* w * (psi) + w_mean * (1-psi)

def generate(args):
    configs =  {'iterations': 0,
                'batch_size': int(args.batch_size),
                'image_width': 256,
                'image_height': 256,
                'mapping_size': 8,
                'latent_size': 512}

    F, G = get_generator(configs)

    F.load_weights(os.path.join(args.cp_src,'mapping_weights.h5'))
    G.load_weights(os.path.join(args.cp_src,'generator_weights.h5'))
    
    w_mean = get_w_mean(F)
    
    generates = []
    for _ in range(configs['batch_size']):
        w = truncate(F(np.random.normal(size = (1, configs['latent_size']))), w_mean, args.psi)
        noise = np.random.normal(size = (1, configs['image_height'], configs['image_width'], 1))#, np.zeros(shape = (1, configs['image_height'], configs['image_width'], 1)), 0.2)
        generates.append(cv2.cvtColor(destandardize_image(G([w, noise])[0]), cv2.COLOR_RGB2BGR))
    

    display_img(generates,out_shape=(256*10,256*10), save_path="preview.jpg")
        
    


def catch_exceptions(args):
    checkpoint_exist = (os.path.isfile("models/checkpoint/generator_weights.h5")
                        or os.path.isfile("models/checkpoint/mapping_weights.h5"))

    return not(checkpoint_exist)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', dest = 'batch_size', type = int, default = 100)
    parser.add_argument('-cs', '--checkpoint-src', dest = 'cp_src', type = str, default = 'models/checkpoint')

    parser.add_argument('-m', '--mode', dest = 'mode', type = int, default = 0)
    parser.add_argument('-psi', dest = 'psi', type = float, default = 0.7)
    args = parser.parse_args()

    if not(catch_exceptions(args)):
        generate(args)
