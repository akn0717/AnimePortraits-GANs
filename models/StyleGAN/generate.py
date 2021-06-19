from utils.plotlib import display_img, plot_multiple_vectors
from StyleGAN_model import *
from utils.imglib import *
import matplotlib.pyplot as plt
import numpy as np
import random
from queue import PriorityQueue
from utils.plotlib import *
import argparse

import os

def interpolate_points(v1, v2, n_steps = 10):
    ratios = np.linspace(0,1,num = n_steps)

    v = []
    for ratio in ratios:
        v.append((1-ratio)*v1 + ratio*v2)

    return v

def repeat(v, nums):
	ans = []
	for _ in range(nums):
		ans.append(v)
	return np.array(ans)


def mode_1(model, E_w, E_noise, batch_size, img_width, img_height, channels, latent_space, beta_1, beta_2):
	E_w = repeat(E_w, batch_size)
	E_noise = repeat(E_noise,batch_size)

	while True:
		noise = np.random.normal(size = (batch_size,img_height, img_width,1))
		z = np.random.normal(size = (batch_size,latent_space))
		
		w = model.F_network.predict(z)

		w = beta_1 * w + (1-beta_1) * E_w
		noise = beta_2 * noise + (1-beta_2) * E_noise

		res = model.Generator.predict([w,noise])

		res = list(((res+1)/2)*255)

		display_img(res, show = 0)


def mode_2(model, E_w, E_noise, batch_size, img_width, img_height, channels, latent_space, beta_1, beta_2, n_steps = 30):
	E_w = repeat(E_w, batch_size)
	E_noise = repeat(E_noise,batch_size)

	noise_1 = np.random.normal(size = (batch_size,img_height, img_width,1))
	z_1 = np.random.normal(size = (batch_size,latent_space))
		
	w_1 = model.F_network.predict(z_1)

	w_1 = beta_1 * w_1 + (1-beta_1) * E_w
	noise_1 = beta_2 * noise_1 + (1-beta_2) * E_noise

	while True:
		
		noise_2 = np.random.normal(size = (batch_size,img_height, img_width,1))
		z_2 = np.random.normal(size = (batch_size,latent_space))

		w_2 = model.F_network.predict(z_2)

		w_2 = beta_1 * w_2 + (1-beta_1) * E_w
		noise_2 = beta_2 * noise_2 + (1-beta_2) * E_noise

		w = interpolate_points(w_1,w_2,n_steps)
		noise = interpolate_points(noise_1,noise_2, n_steps)
		for i in range(len(w)):
			res = model.Generator.predict([w[i],noise[i]])

			res = list(((res+1)/2)*255)
			display_img(res,show = 1)
		
		w_1 = w_2
		noise_1 = noise_2


def generate(args):

	beta_1 = args.beta_1
	beta_2 = args.beta_2
	img_width, img_height = 256, 256
	batch_size = int(args.batch_size)
	latent_space = 512

	model = StyleGAN(batch_size = batch_size, img_height = img_height, img_width = img_width, channels = 3, path = args.model_path)

	model.F_network.load_weights("trained_model/Mapping_Network.h5")
	model.Generator.load_weights("trained_model/Generator.h5")


	E_w = np.zeros(shape = (512,))
	E_noise = np.zeros(shape = (img_width,img_height,1))

	for _ in range(1000):
		z = np.random.normal(size = (1,latent_space))
		E_w = E_w + model.F_network.predict(z)[0]

	E_w /= 1000

	mode_2(model, E_w, E_noise, batch_size, img_width, img_height, 3, latent_space, beta_1, beta_2)
	return

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-b', '--batch_size', dest = 'batch_size', default = 1)
	parser.add_argument('-m', '--model-path', dest = 'model_path', default = 'trained_model')
	parser.add_argument('-b1', '--beta_1', dest = 'beta_1', type = float, default = 0.7)
	parser.add_argument('-b2', '--beta_2', dest = 'beta_2', type = float, default = 0.2)
	args = parser.parse_args()
	
	generate(args)