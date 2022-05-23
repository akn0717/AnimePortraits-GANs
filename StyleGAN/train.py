import argparse
from genericpath import exists
import json
import os
import shutil

import cv2
from models.discriminator import get_discriminator
from models.generator import get_FG, get_generator
from models.trainer import Trainer, save_checkpoint

from utils.batchgen import BatchGen
from utils.imglib import destandardize_image
from utils.plotlib import display_img, plot_multiple_vectors
from collections import deque 

import time

def sort_key(s):
    num = -1
    if (len(s.split('_'))>1):
        num = int(s.split('_')[1])
    return num

def train(args):

    #Hyperparameters
    lr = 0.0001
    n_critic = 5 if args.loss == 0 else 1
    print(n_critic)
    num_images = 25

    #Prepare configs
    configs =  {'iterations': 0,
                'batch_size': int(args.batch_size),
                'image_width': 256,
                'image_height': 256,
                'mapping_size': 8,
                'latent_size': int(args.latent_size)}

    H, W, C = 256, 256, 3

    #Batch Sampler
    ImageGen = BatchGen(args.data_src, (H, W), args.batch_size, args.latent_size)
    data_size = ImageGen.get_size()
    print("Data size: ",data_size," samples")

    #Prepare models
    trainer = Trainer(lr = lr, loss_fn = args.loss)
    trainer.set_BatchGen(ImageGen)

    F, G = get_generator(configs)
    D = get_discriminator()
    FG = get_FG(configs, F, G)

    trainer.FG = FG
    trainer.D = D
    trainer.G = G
    trainer.F = F
    FG.summary()
    D.summary()

    dq = deque()

    if not(args.train_new):
        checkpoint_list = [f for f in os.listdir(args.cp_src) if len(f.split('_')) > 1 and f.split('_')[0] == 'checkpoint']
        checkpoint_list.sort(key = sort_key)
        dq = deque(checkpoint_list)
        print("Loading ", dq[-1], "...",sep = "")
        if (len(dq) >= 1):
            trainer.load(os.path.join(args.cp_src, checkpoint_list[-1]))
        else:
            trainer.load(os.path.join(args.cp_src, 'checkpoint'))
    else:
        trainer.init_visualization(num_images)
        result = destandardize_image(trainer.get_preview(FG, num_images, random = False))
        display_img(list(result), save_path = os.path.join(args.cp_src,"Preview_0.png"))

    run_time = 0
    
    while (True):
        start_time = time.time()
        for _ in range(n_critic):
			##Train Discriminator
            reals, z, noise = ImageGen.next_batch_Discriminator()
            D_loss = trainer.discriminator_step([reals, z, noise])
        
        ##Train Generator
        z, noise = ImageGen.next_batch_Generator()
        G_loss = trainer.generator_step([z, noise])

        trainer.d_loss.append(D_loss)
        trainer.g_loss.append(G_loss)

        run_time += (time.time() - start_time)
        if trainer.iterations%int(args.log_iter)==0:
            print('epoch: ', (trainer.iterations//(data_size//args.batch_size))+1,' iterations: ',trainer.iterations, ' ', trainer.iterations*int(args.batch_size)//1000,'kimg',' loss D: ',D_loss,' loss G: ',G_loss,' [',1.*run_time/int(args.log_iter),'s]',sep = '')
            result = destandardize_image(trainer.get_preview(FG, 25))
            display_img(list(result), save_path = os.path.join(args.cp_src,'Preview.jpg'))
            plot_multiple_vectors([trainer.d_loss,trainer.g_loss], title = 'loss', xlabel='iterations', legends = ['Discriminator Loss', 'Generator Loss'], save_path = os.path.join(args.cp_src,'loss.png'))
            run_time = 0
            

        if trainer.iterations%int(args.cp_iter)==0:
            print("Saving...", end = '')
            trainer.save(os.path.join(args.cp_src, 'checkpoint_' + str(trainer.iterations)), configs)
            plot_multiple_vectors([trainer.d_loss,trainer.g_loss], title = 'loss', xlabel='iterations', legends = ['Discriminator Loss', 'Generator Loss'], save_path = os.path.join(args.cp_src, 'checkpoint_'+str(trainer.get_num_iteration())+'/loss.png'))
            dq.append('checkpoint_'+str(trainer.iterations))
            while (len(dq) > 5):
                top = dq[0]
                shutil.rmtree(os.path.join(args.cp_src,top))
                dq.popleft()
            
            trainer.save(os.path.join(args.cp_src, 'checkpoint'), configs)
            plot_multiple_vectors([trainer.d_loss,trainer.g_loss], title = 'loss', xlabel='iterations', legends = ['Discriminator Loss', 'Generator Loss'], save_path = os.path.join(args.cp_src, 'checkpoint/loss.png'))
            print("Done!")

        if args.footstep!=0 and trainer.iterations%int(args.footstep)==0:
            result = destandardize_image(trainer.get_preview(FG, num_images, random = False))
            display_img(list(result), save_path = os.path.join(args.cp_src,"Preview_"+str(trainer.iterations)+".png"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--new', action = 'store_true', dest = 'train_new', default = False)

    parser.add_argument('-b', '--batch-size', dest = 'batch_size', type = int, default = 4)
    parser.add_argument('-ls', '--latent-size', dest = 'latent_size', type = int, default = 512)
    parser.add_argument('-loss', '--loss-function', help = 'loss function: 0 for WGAN-GP, 1 for LSGAN (default: 0)', dest = 'loss', type = int, default = 0)

    parser.add_argument('-c', '--checkpoint-iteration', dest = 'cp_iter', type = int, default = 100)
    parser.add_argument('-cs', '--checkpoint-source', dest = 'cp_src', type = str, default = 'models/checkpoints')
    parser.add_argument('-ds', '--data-source', dest = 'data_src', type = str, default = 'dataset/d1k.h5')

    parser.add_argument('-l', '--log', dest = 'log_iter', type = int, default = 10)
    parser.add_argument('-f', '--footstep', dest = 'footstep', type = int, default = 0)
    args = parser.parse_args()

    train(args)
