import matplotlib.pyplot as plt
import numpy as np
import random

from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Dropout, Flatten, Reshape, Conv2DTranspose, Input
from keras.layers import Activation,LeakyReLU, BatchNormalization, UpSampling2D
from keras.optimizers import adam

from collections import deque

import os


img_width, img_height = 64, 64

latent_space = 100

data_dir = 'Dataset'
if not os.path.isdir('Model'): os.mkdir('Model')
G_dir = 'Model/Generator.h5'
D_dir = 'Model/Discriminator.h5'
batch_size = 32
cnt = 0
#############image processing
img_file = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
img_array = []
for i in img_file:
	#print("Loading:",cnt,"/",len(img_file))
	cnt+=1
	#if (cnt==10000):
	#	break
	img_load = image.load_img(os.path.join(data_dir,i),target_size = (img_width,img_height))
	img_load = np.reshape(img_load,(img_width,img_height,3))
	img_load = img_load/127.5 - 1
	img_array.append(img_load)
########Build Neural Network
##Build Generator

x = Input(shape = (latent_space,))
Gen_hidden = Dense(8192) (x)
Gen_hidden = Reshape((4,4,512)) (Gen_hidden)
Gen_hidden = BatchNormalization(momentum = 0.8) (Gen_hidden)
Gen_hidden = LeakyReLU(0.2) (Gen_hidden)
Gen_hidden = Conv2DTranspose(256,(5,5),strides = (2,2),padding = 'same') (Gen_hidden)
Gen_hidden = BatchNormalization(momentum = 0.8) (Gen_hidden)
Gen_hidden = LeakyReLU(0.2) (Gen_hidden)
Gen_hidden = Conv2DTranspose(128,(5,5),strides = (2,2),padding = 'same') (Gen_hidden)
Gen_hidden = BatchNormalization(momentum = 0.8) (Gen_hidden)
Gen_hidden = LeakyReLU(0.2) (Gen_hidden)
Gen_hidden = Conv2DTranspose(64,(5,5),strides = (2,2),padding = 'same') (Gen_hidden)
Gen_hidden = BatchNormalization(momentum = 0.8) (Gen_hidden)
Gen_hidden = LeakyReLU(0.2) (Gen_hidden)
x_out = Conv2DTranspose(3,(5,5),strides = (2,2),padding = 'same',activation = 'tanh') (Gen_hidden)
Generator = Model(x,x_out)
Generator.compile(loss = 'binary_crossentropy', optimizer = adam())

#if os.path.isfile(G_dir): Generator.load_weights(G_dir)

##Build Discriminator
x = Input(shape = (img_width,img_height,3))
D_hidden = Conv2D(64,(5,5),strides = (2,2),padding = 'same') (x)
D_hidden = LeakyReLU(0.2) (D_hidden)
D_hidden = Conv2D(128,(5,5),strides = (2,2),padding = 'same') (D_hidden)
D_hidden = LeakyReLU(0.2) (D_hidden)
D_hidden = Conv2D(256,(5,5),strides = (2,2),padding = 'same') (D_hidden)
D_hidden = LeakyReLU(0.2) (D_hidden)
D_hidden = Conv2D(512,(5,5),strides = (2,2),padding = 'same') (D_hidden)
D_hidden = LeakyReLU(0.2) (D_hidden)
D_hidden = Flatten() (D_hidden)
x_out = Dense(1,activation = 'sigmoid') (D_hidden)
Discriminator = Model(x,x_out)
Discriminator.compile(loss = 'binary_crossentropy', optimizer = adam(0.0002,0.5))

#if os.path.isfile(D_dir): Discriminator.load_weights(D_dir)

#print(Discriminator.predict(img_array[0]/127.5 - 1))
#quit()
Discriminator.trainable = False
GD = Sequential()
GD.add(Generator)
GD.add(Discriminator)
GD.compile(loss = 'binary_crossentropy', optimizer = adam(0.00010,0.5))

##########################################################################################
frame = deque()
g_loss = deque()
d_loss = deque()

img_fig = plt.figure(figsize = (19,10))
graph_fig = plt.figure(figsize = (7,2))
loss_fig = graph_fig.add_subplot(1,1,1)

g = []
n = int(np.sqrt(batch_size))
for i in range(n*n):
	g.append(img_fig.add_subplot(n,n,i+1))

########Train
Answer_Y = np.ones((batch_size,1))
Answer_N = np.zeros((batch_size,1))
noise = np.asarray(np.random.normal(size = (batch_size,latent_space)))
cnt = 0
G_loss = 0
D_loss = 0
while (True):
	cnt+=1
	real_image = random.sample(img_array, batch_size)
	z = np.asarray(np.random.normal(size = (batch_size,latent_space)))
	fake_image = Generator.predict(z)
	###Train Discriminator
	img = np.concatenate((real_image,fake_image),axis = 0)
	label = np.concatenate((Answer_Y,Answer_N),axis = 0)
	Discriminator.trainable = True
	D_loss = Discriminator.train_on_batch(img,label)
	###Train Generator
	z = np.asarray(np.random.normal(size = (batch_size,latent_space)))
	Discriminator.trainable = False
	G_loss = GD.train_on_batch(z, Answer_Y)
	if (cnt%10==0):
		d_loss.append(D_loss)
		g_loss.append(G_loss)
		frame.append(cnt)
	if cnt%1000==0:
		print('epochs: ',cnt,' loss D: ',D_loss,' loss G: ',G_loss)
	if cnt%1000==0:
		result = (Generator.predict(noise)+1)/2
		for i in range(len(g)):
			g[i].imshow(result[i])
		loss_fig.plot(frame,d_loss,'b-')
		loss_fig.plot(frame,g_loss,'g-')
		img_fig.savefig("Preview.jpg")
		graph_fig.savefig('loss.jpg')
		for i in range(len(g)):
			g[i].cla()
		loss_fig.cla()
	if cnt%5000==0:
		Discriminator.save_weights(D_dir)
		Generator.save_weights(G_dir)
		print('Saving...')