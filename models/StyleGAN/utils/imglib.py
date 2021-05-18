from keras.preprocessing import image
import os
import numpy as np

def load_images(data_dir, W, H):
    img_list = []
    img_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    for f in img_files:
        img_load = image.load_img(os.path.join(data_dir,f),target_size = (W,H))
        img_load = np.reshape(img_load,(W,H,3))
        img_list.append(img_load)
    return img_list

def load_sampling(data_dir,name_files, W, H):
    img_list = []
    img_files = name_files
    for f in img_files:
        img_load = image.load_img(os.path.join(data_dir,f),target_size = (W,H))
        img_load = np.reshape(img_load,(W,H,3))
        img_list.append(img_load)
    return img_list