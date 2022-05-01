import argparse
from json import JSONEncoder
import json
import os
from tkinter import Image
from matplotlib.font_manager import json_dump
from models.discriminator import get_discriminator
from models.generator import get_FG, get_generator
from models.trainer import Trainer, save_checkpoint

from utils.batchgen import BatchGen
from utils.imglib import destandardize_image
from utils.plotlib import display_img, plot_multiple_vectors

def train(args):

    #Hyperparameters
    lr = 0.0001
    n_critic = 5
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
    trainer = Trainer(lr = lr)
    trainer.set_BatchGen(ImageGen)

    F, G = get_generator(configs)
    D = get_discriminator()
    FG = get_FG(configs, F, G)

    trainer.FG = FG
    trainer.D = D
    FG.summary()
    D.summary()
    
    if not(args.train_new):
        trainer.load_optimizers(args.cp_src, FG, D)
        F.load_weights(os.path.join(args.cp_src,'mapping_weights.h5'))
        G.load_weights(os.path.join(args.cp_src,'generator_weights.h5'))
        D.load_weights(os.path.join(args.cp_src,'discriminator_weights.h5'))
        trainer.load(args.cp_src)
    else:
        trainer.init_visualization(num_images)
        result = destandardize_image(trainer.get_preview(FG, num_images, random = False))
        display_img(list(result), save_path = os.path.join(args.cp_src,"Preview_0.png"))

        
    while (True):
        
        for _ in range(n_critic):
			##Train Discriminator
            reals, z, noise = ImageGen.next_batch_Discriminator()
            D_loss = trainer.discriminator_step([reals, z, noise])
        
        ##Train Generator
        z, noise = ImageGen.next_batch_Generator()
        G_loss = trainer.generator_step([z, noise])

        trainer.d_loss.append(D_loss)
        trainer.g_loss.append(G_loss)

        iterations = trainer.get_num_iteration()
        
        if iterations%int(args.log_iter)==0:
            print('epoch: ', (iterations//(data_size//args.batch_size))+1,' iterations: ',iterations,' loss D: ',D_loss,' loss G: ',G_loss)
            result = destandardize_image(trainer.get_preview(FG, 9))
            display_img(list(result), save_path = os.path.join(args.cp_src,'Preview.jpg'))
            plot_multiple_vectors([trainer.d_loss,trainer.g_loss], title = 'loss', xlabel='iterations', legends = ['Discriminator Loss', 'Generator Loss'], save_path = os.path.join(args.cp_src,'loss.png'))

        if iterations%int(args.cp_iter)==0:
            print("Saving...", end = '')
            trainer.save(args.cp_src)
            save_checkpoint(F, os.path.join(args.cp_src, "mapping_weights.h5"))
            save_checkpoint(G, os.path.join(args.cp_src, "generator_weights.h5"))
            save_checkpoint(D, os.path.join(args.cp_src, "discriminator_weights.h5"))
            configs['iterations'] = trainer.get_num_iteration()
            with open(os.path.join(args.cp_src,"log.json"), "w") as f:
                json.dump(configs, f)
            print("Done!")

        if args.footstep!=0 and iterations%int(args.footstep)==0:
            result = destandardize_image(trainer.get_preview(FG, num_images, random = False))
            display_img(list(result), save_path = os.path.join(args.cp_src,"Preview_"+str(iterations)+".png"))

def catch_exceptions(args):
    train_new = args.train_new
    checkpoint_exist = True

    if (train_new):
        os.makedirs(args.cp_src, exist_ok = True)
    else:
        checkpoint_exist = (os.path.isfile(os.path.join(args.cp_src,"generator_weights.h5"))
                            or os.path.isfile(os.path.join(args.cp_src,"discriminator_weights.h5"))
                            or os.path.isfile(os.path.join(args.cp_src,"mapping_weights.h5"))
                            or os.path.isfile(os.path.join(args.cp_src,"optimizer_G_checkpoint.pkl"))
                            or os.path.isfile(os.path.join(args.cp_src,"optimizer_D_checkpoint.pkl"))
                            or os.path.isfile(os.path.join(args.cp_src,"log.json"))
                            or os.path.isfile(os.path.join(args.cp_src,"training_checkpoint.h5")))

    dataset_exist = os.path.isfile(args.data_src)

    return not(checkpoint_exist and dataset_exist)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--new', action = 'store_true', dest = 'train_new', default = False)

    parser.add_argument('-b', '--batch-size', dest = 'batch_size', type = int, default = 4)
    parser.add_argument('-ls', '--latent-size', dest = 'latent_size', type = int, default = 512)


    parser.add_argument('-c', '--checkpoint-iteration', dest = 'cp_iter', type = int, default = 100)
    parser.add_argument('-cs', '--checkpoint-source', dest = 'cp_src', type = str, default = 'models/checkpoint')
    parser.add_argument('-ds', '--data-source', dest = 'data_src', type = str, default = 'dataset/d1k.h5')

    parser.add_argument('-l', '--log', dest = 'log_iter', type = int, default = 10)
    parser.add_argument('-f', '--footstep', dest = 'footstep', type = int, default = 0)
    args = parser.parse_args()

    if not(catch_exceptions(args)):
        train(args)
