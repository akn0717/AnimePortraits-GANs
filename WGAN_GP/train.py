from models.WGAN_GP import *
from utils.imglib import *
import matplotlib.pyplot as plt
import numpy as np
import random

from collections import deque

import os


img_width, img_height = 64, 64

data_dir = 'Dataset'
G_dir = 'Model/Generator.h5'
D_dir = 'Model/Discriminator.h5'
batch_size = 32
latent_space = 100
cnt = 0

frame = deque()
g_loss = deque()
d_loss = deque()

img_fig = plt.figure(figsize = (15,10))
graph_fig = plt.figure(figsize = (7,2))
loss_fig = graph_fig.add_subplot(1,1,1)

g = []
n = int(np.sqrt(batch_size))
for i in range(n*n):
	g.append(img_fig.add_subplot(n,n,i+1))

########Train
Answer_Y = np.ones((batch_size,1))
Answer_N = np.zeros((batch_size,1))
noise = np.random.normal(size = (batch_size,latent_space))
cnt = 0
D_loss = 0

model = WGANGP_model()


dataset = load_images(data_dir, img_width, img_height)
n_critic = 5

while (True):
	cnt+=1
	for _ in range(n_critic):
		model.reals = np.array(random.sample(dataset, batch_size))/127.5 - 1
		model.noise = np.random.normal(size = (batch_size,latent_space))
		##Train Discriminator
		model.Discriminator.trainable = True
		D_loss = model.train_on_batch_D()

    ###Train Generator
	model.noise = np.random.normal(size = (batch_size,latent_space))
	model.Discriminator.trainable = False
	G_loss = model.train_on_batch_G()

	d_loss.append(D_loss)
	frame.append(cnt)

	if (cnt%100==0):
		print('epochs: ',cnt,' loss D: ',-D_loss,' loss G',G_loss)

	if cnt%100==0:
		result = (model.Generator.predict(noise)+1)/2
		for i in range(len(g)):
			g[i].imshow(result[i])
		loss_fig.plot(frame,d_loss,'b-')
		img_fig.savefig("Preview"+str(cnt)+".jpg")
		graph_fig.savefig('loss.jpg')
		for i in range(len(g)):
			g[i].cla()
		loss_fig.cla()
	
	if cnt%1000==0:
		print("Saving...")
		model.Discriminator.save_weights('Discriminator.h5')
		model.Generator.save_weights('Generator.h5')