from BatchGen import BatchGen
from utils.plotlib import display_img, plot_multiple_vectors
from StyleGAN_model import *
from utils.imglib import *

from utils.plotlib import *
import argparse

import os

def train(args):

	img_width, img_height = 256, 256

	batch_size = int(args.batch_size)
	latent_space = 512

	ImageGen = BatchGen(args.dataset, (img_width, img_height), batch_size, latent_space)

	########Train
	
	D_loss = 0

	model = StyleGAN(batch_size = batch_size, img_height = img_height, img_width = img_width, channels = 3, ImageGen = ImageGen, train_new = args.train_new, path = args.model_path, lr = 5e-6)

	if (args.train_new == True):
		model.z_visual, model.noise_visual = ImageGen.next_batch_Generator()

	n_critic = 5

	z, noise = ImageGen.next_batch_Generator()

	data_size = ImageGen.get_size()
	print("Data size: ",data_size," samples")

	iterations = model.get_iteration()
	result = ((model.FG.predict([model.z_visual, model.noise_visual])+1)/2)*255
	display_img(list(result), save_path = os.path.join(args.model_path,'Preview.jpg'))

	if (args.history and iterations == 1):
		display_img(list(result), save_path = os.path.join(args.model_path,"Preview_0.png"))

	while (True):
		iterations = model.get_iteration()
		model.Discriminator.trainable = True
		for _ in range(n_critic):
			##Train Discriminator
			model.reals, model.z, model.noise = ImageGen.next_batch_Discriminator()
			D_loss = model.train_on_batch_D()

		###Train Generator
		model.z, model.noise = ImageGen.next_batch_Generator()
		model.Discriminator.trainable = False
		G_loss = model.train_on_batch_G()

		model.d_loss.append(D_loss)
		model.g_loss.append(G_loss)


		if iterations%int(args.detail_iteration)==0:
			print('epoch: ', (iterations//(data_size//batch_size))+1,' iterations: ',iterations,' loss D: ',D_loss,' loss G: ',G_loss)
			z, noise = ImageGen.next_batch_Generator()
			result = ((model.FG.predict([z, noise])+1)/2)*255
			display_img(list(result), save_path = os.path.join(args.model_path,'Preview.jpg'))
			plot_multiple_vectors([model.d_loss,model.g_loss], title = 'loss', xlabel='iterations', legends = ['Discriminator Loss', 'Generator Loss'], save_path = os.path.join(args.model_path,'loss.png'))

		if iterations%int(args.save_iteration)==0:
			model.save_model(args.model_path)

		if iterations%int(args.preview_iteration)==0:
			if (args.history):
				result = ((model.FG.predict([model.z_visual, model.noise_visual])+1)/2)*255
				display_img(list(result), save_path = os.path.join(args.model_path,"Preview_"+str(iterations)+".png"))
	
	return

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--new', action = 'store_true', dest = 'train_new', default = False)
	parser.add_argument('-b', '--batch_size', dest = 'batch_size', default = 2)
	parser.add_argument('-de', '--iteration-detail', dest = 'detail_iteration', default = 10)
	parser.add_argument('-p', '--iteration-preview', dest = 'preview_iteration', default = 100)
	parser.add_argument('-s', '--iteration-save', dest = 'save_iteration', default = 100)
	parser.add_argument('-d', '--data', dest = 'dataset', default = 'Dataset/d1k.h5')
	parser.add_argument('-history', '--preview-history', action = 'store_true', dest = 'history', default = True)
	parser.add_argument('-m', '--model-path', dest = 'model_path', default = 'trained_model')
	args = parser.parse_args()
	
	train(args)