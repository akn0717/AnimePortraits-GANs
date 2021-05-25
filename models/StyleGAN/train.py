from utils.plotlib import display_img, plot_multiple_vectors
from StyleGAN_model import *
from utils.imglib import *
import matplotlib.pyplot as plt
import numpy as np
import random

from utils.plotlib import *
import argparse

import os

def train(args):

	img_width, img_height = 256, 256

	data_dir = args.dataset
	batch_size = int(args.batch_size)
	latent_space = 512
	cnt = 0

	g_loss = []
	d_loss = []

	########Train
	z = np.random.normal(size = (batch_size,latent_space))
	noise = np.random.normal(size = (batch_size, img_height, img_width,1))
	
	D_loss = 0


	model = StyleGAN(batch_size = batch_size, img_height = img_height, img_width = img_width, channels = 3, path = args.model_path)
	
	if (args.train_new == False):
		model.load_model(args.model_path)
	
	img_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
	n_critic = 5

	
	while (True):
		model.iteration+=1
		for _ in range(n_critic):
			real_samples = random.sample(img_files, batch_size)
			model.reals = np.array(load_sampling(data_dir, real_samples, img_height, img_width))/127.5 - 1
			model.z = np.random.normal(size = (batch_size,latent_space))
			model.noise = np.random.normal(size = (batch_size, img_height, img_width,1))
			##Train Discriminator
			model.Discriminator.trainable = True

			D_loss = model.train_on_batch_D()

		###Train Generator
		model.z = np.random.normal(size = (batch_size,latent_space))
		model.noise = np.random.normal(size = (batch_size, img_height, img_width,1))
		model.Discriminator.trainable = False
		G_loss = model.train_on_batch_G()

		d_loss.append(-D_loss)
		g_loss.append(G_loss)

		if model.iteration%int(args.detail_iteration)==0:
			print('epoch: ', int(model.iteration/model.batch_size),'iterations: ',model.iteration,' loss D: ',-D_loss,' loss G: ',G_loss)

		if model.iteration%int(args.save_iteration)==0:
			model.save_model(args.model_path)

		if model.iteration%int(args.preview_iteration)==0:
			z_sample = np.random.normal(size = (batch_size,latent_space))
			noise_sample = np.random.normal(size = (batch_size, img_height, img_width,1))
			result = ((model.Generator.predict([z_sample, noise_sample])+1)/2)*255
			display_img(list(result), save_path = 'Preview.jpg')
			plot_multiple_vectors([d_loss,g_loss], title = 'loss', xlabel='iterations', legends = ['Discriminator Loss', 'Generator Loss'], save_path = 'loss')
		
		if model.iteration%1000==0:
			result = ((model.Generator.predict([z, noise])+1)/2)*255
			display_img(list(result), save_path = '/content/drive/MyDrive/Preview'+str(model.iteration)+'.jpg')
	
	return

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--new', action = 'store_true', dest = 'train_new', default = False)
	parser.add_argument('-b', '--batch_size', dest = 'batch_size', default = 16)
	parser.add_argument('-d', '--iteration-detail', dest = 'detail_iteration', default = 10)
	parser.add_argument('-p', '--iteration-preview', dest = 'preview_iteration', default = 100)
	parser.add_argument('-s', '--iteration-save', dest = 'save_iteration', default = 100)
	parser.add_argument('-d', '--data', dest = 'dataset', default = 'Dataset')
	parser.add_argument('-m', '--model-path', dest = 'model_path', default = 'trained_model')
	args = parser.parse_args()
	
	train(args)