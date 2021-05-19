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
	batch_size = 32
	latent_space = 512
	cnt = 0

	g_loss = []
	d_loss = []

	########Train
	noise = np.random.normal(size = (batch_size,latent_space))

	D_loss = 0


	model = StyleGAN(batch_size = batch_size, img_height = img_height, img_width = img_width, channels = 3, path = args.model_path)
	
	if (args.train_new == False):
		model.load_model(args.model_path)
	
	model.save_model(args.model_path)
	img_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
	n_critic = 5

	while (True):
		model.epoch+=1
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

		if model.epoch%int(args.detail_epoch)==0:
			print('epochs: ',model.epoch,' loss D: ',-D_loss,' loss G',G_loss)

		if model.epoch%int(args.preview_epoch)==0:
			result = ((model.Generator.predict([model.z, model.noise])+1)/2)*255
			display_img(list(result), save_path = 'Preview.jpg')
			plot_multiple_vectors([d_loss,g_loss], title = 'loss', xlabel='epochs', legends = ['Discriminator Loss', 'Generator Loss'], save_path = 'loss')
		
		if model.epoch%int(args.save_epoch)==0:
			print("Saving...")
			model.save_model(args.model_path)
	
	return

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--new', action = 'store_true', dest = 'train_new', default = False)
	parser.add_argument('-ed', '--epoch-detail', dest = 'detail_epoch', default = 10)
	parser.add_argument('-ep', '--epoch-preview', dest = 'preview_epoch', default = 100)
	parser.add_argument('-es', '--epoch-save', dest = 'save_epoch', default = 100)
	parser.add_argument('-d', '--data', dest = 'dataset', default = 'Dataset')
	parser.add_argument('-m', '--model-path', dest = 'model_path', default = 'trained_model')
	args = parser.parse_args()
	
	train(args)