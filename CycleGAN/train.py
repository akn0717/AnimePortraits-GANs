import argparse
import json
import os
import cv2

import numpy as np
from models.batchgen import BatchGen

from models.models import D_step, G_step, get_discriminator, get_generator, get_optimizer, load_checkpoint, save_checkpoint

from utils.imglib import destandardize_image
from utils.plotlib import display_img, plot_multiple_vectors

import time



def train(args):

    #Hyperparameters
    lr_base = 2e-6
    num_images = 25

    #Prepare configs
    img_shape = (256, 256, 3)
    batch_size = int(args.batch_size)

    #Batch Sampler
    ImageGen = BatchGen(args.A_src, args.B_src)
    A_size, B_size = ImageGen.get_size()
    print("num images in A: ",A_size)
    print("num images in B: ",B_size)

    #Prepare models
    G_A = get_generator(img_shape)
    D_A = get_discriminator(img_shape)
    G_B = get_generator(img_shape)
    D_B = get_discriminator(img_shape)
    optimizer_A = get_optimizer(lr_base = lr_base)
    optimizer_B = get_optimizer(lr_base = lr_base)

    if not(args.train_new):
        load_checkpoint(G_A, os.path.join(args.cp_src, "GA_weights.h5"))
        load_checkpoint(D_A, os.path.join(args.cp_src, "DA_weights.h5"))
        load_checkpoint(G_B, os.path.join(args.cp_src, "GB_weights.h5"))
        load_checkpoint(D_B, os.path.join(args.cp_src, "DB_weights.h5"))

    G_A.summary()
    D_A.summary()

    run_time = 0

    D_A_loss = []
    G_A_loss = []
    D_B_loss = []
    G_B_loss = []

    while (True):
        start_time = time.time()
        # Train Discriminator
        A_samples = ImageGen.get_A_samples(batch_size//2, img_shape)
        B_samples = ImageGen.get_B_samples(batch_size//2, img_shape)
        A_generates = G_A.predict(B_samples)
        B_generates = G_B.predict(A_samples)
        x_A = np.concatenate([A_samples, A_generates], axis = 0)
        x_B = np.concatenate([B_samples, B_generates], axis = 0)
        d_A_loss = D_step(D_A, optimizer_A, A_samples, A_generates)
        d_B_loss = D_step(D_B, optimizer_B, B_samples, B_generates)
        
        # Train Generator
        A_samples = ImageGen.get_A_samples(batch_size, img_shape)
        B_samples = ImageGen.get_B_samples(batch_size, img_shape)

        g_A_loss = G_step(G_A, G_B, D_A, optimizer_A, B_samples, A_samples)
        g_B_loss = G_step(G_B, G_A, D_B, optimizer_B, A_samples, B_samples)

        G_A_loss.append(g_A_loss)
        D_A_loss.append(d_A_loss)
        G_B_loss.append(g_B_loss)
        D_B_loss.append(d_B_loss)

        iterations = np.int(optimizer_A.iterations) // 2
        run_time += (time.time() - start_time)
        if iterations%int(args.log_iter)==0:
            print('iterations: ',iterations)
            print(' loss D_A: ',d_A_loss,' loss G_A: ',g_A_loss)
            print(' loss D_B: ',d_B_loss,' loss G_B: ',g_B_loss)
            print('[',1.*run_time/int(args.log_iter),'s/iter]')
            A_samples = ImageGen.get_A_samples(3, img_shape)
            B_samples = ImageGen.get_B_samples(3, img_shape)
            A_generated = G_A.predict(B_samples)
            B_generated = G_B.predict(A_samples)
            B_reconstructed = G_B(A_generated)
            A_reconstructed = G_A(B_generated)
            B_view = np.concatenate([B_samples, A_generated, B_reconstructed], axis = 0)
            A_view = np.concatenate([A_samples, B_generated, A_reconstructed], axis = 0)
            A_view = list(destandardize_image(A_view))
            B_view = list(destandardize_image(B_view))

            display_img(list(A_view), save_path = os.path.join(args.cp_src,'Preview_A.jpg'))
            display_img(list(B_view), save_path = os.path.join(args.cp_src,'Preview_B.jpg'))

            plot_multiple_vectors([D_A_loss,G_A_loss], title = 'loss', xlabel='iterations', legends = ['Discriminator Loss', 'Generator Loss'], save_path = os.path.join(args.cp_src,'loss_A.png'))
            plot_multiple_vectors([D_B_loss,G_B_loss], title = 'loss', xlabel='iterations', legends = ['Discriminator Loss', 'Generator Loss'], save_path = os.path.join(args.cp_src,'loss_B.png'))
            print("##########################################################################")
            run_time = 0

        if iterations%int(args.cp_iter)==0:
            print("Saving...", end = '')
            save_checkpoint(G_A, os.path.join(args.cp_src, "GA_weights.h5"))
            save_checkpoint(D_A, os.path.join(args.cp_src, "DA_weights.h5"))
            save_checkpoint(G_B, os.path.join(args.cp_src, "GB_weights.h5"))
            save_checkpoint(D_B, os.path.join(args.cp_src, "DB_weights.h5"))
            print("Done!")

        # if args.footstep!=0 and iterations%int(args.footstep)==0:
        #     result = destandardize_image(trainer.get_preview(FG, num_images, random = False))

        #     display_img(list(result), save_path = os.path.join(args.cp_src,"Preview_"+str(iterations)+".png"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--new', action = 'store_true', dest = 'train_new', default = False)

    parser.add_argument('-b', '--batch-size', dest = 'batch_size', type = int, default = 4)

    parser.add_argument('-c', '--checkpoint-iteration', dest = 'cp_iter', type = int, default = 1000)
    parser.add_argument('-cs', '--checkpoint-source', dest = 'cp_src', type = str, default = 'models/checkpoint')
    parser.add_argument('-A', '--A-source', dest = 'A_src', type = str, default = 'dataset/A.h5')
    parser.add_argument('-B', '--B-source', dest = 'B_src', type = str, default = 'dataset/B.h5')

    parser.add_argument('-l', '--log', dest = 'log_iter', type = int, default = 100)
    parser.add_argument('-f', '--footstep', dest = 'footstep', type = int, default = 0)
    args = parser.parse_args()

    train(args)
