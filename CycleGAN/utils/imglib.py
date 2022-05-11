import os
import numpy as np
import cv2

def h5_load_image(h5, key):
    return h5['key'].value

def h5_load_images(h5, keys):
    img_list = []
    for key in keys:
        img_list.append(h5[key].value)
    return img_list

def load_image(data_dir, W, H):
    img_load = cv2.imread(data_dir)

    return img_load

def load_images(data_dir, W, H):
    img_list = []
    img_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    for f in img_files:
        img_load = cv2.imread(os.path.join(data_dir,f))
        img_list.append(img_load)
    return img_list

def load_sampling(data_dir,name_files, W, H):
    img_list = []
    img_files = name_files
    for f in img_files:
        img_load = cv2.imread(os.path.join(data_dir,f))
        img_load = cv2.resize(img_load, (H,W))
        img_list.append(img_load)
    return img_list

def standardize_image(imgs):
    return (np.array(imgs) / 127.5) - 1

def destandardize_image(imgs):
    return np.array((np.array(imgs) + 1) * 127.5, dtype = np.int)

def rand_crop(img, crop_shape):
    img = np.array(img)
    H, W = img.shape[0], img.shape[1]

    x = np.random.randint(0, H-crop_shape[0] + 1)
    y = np.random.randint(0, W-crop_shape[1] + 1)

    return img[x:x+crop_shape[0], y:y+crop_shape[1]]