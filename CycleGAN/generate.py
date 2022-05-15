
import argparse
import os
import cv2

import numpy as np
from models.models import get_generator
from utils.imglib import destandardize_image, load_image, standardize_image

from utils.plotlib import display_img

def padding_img(img, out_shape):
    if (img.shape[0] > out_shape[0] or img.shape[1] > out_shape[1]):
        print("ERROR: padding is invalid")
        return

    if (img.shape[0] == out_shape[0] and img.shape[1] == out_shape[1]):
        return img

    out_img = np.zeros(out_shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                out_img[i,j,k] = img[i,j,k]
    return out_img


def resize(img, out_shape):
    return cv2.resize(img, out_shape)

def cropping(img, out_shape):
    H, W, C = np.array(img).shape
    out_H, out_W, out_C = out_shape
    out_list = []
    x_cnt, y_cnt = 0, 0
    for y in range(0, H, out_H):
        y_cnt += 1
        for x in range(0, W, out_W):
            out_list.append(img[y : min(H, y + out_H), x : min(W, x + out_W), :])
    return list(out_list), int(W // out_W)+1, y_cnt

def rejoin(crop_list):
    pass

def process(model, src_path, dst_path):
    A = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
    for idx in range(len(A)):
        frame_crops, out_W, out_H = cropping(load_image(os.path.join(src_path, A[idx])), out_shape=(256, 256, 3))
        A_pred = []
        for frame_idx in range(len(frame_crops)):
            x = np.reshape(frame_crops[frame_idx], (1, frame_crops[frame_idx].shape[0], frame_crops[frame_idx].shape[1], frame_crops[frame_idx].shape[2]))
            A_pred.append(destandardize_image(model.predict(standardize_image(x))[0]))
        
        display_img(list(A_pred), shape = (out_H,out_W), save_path=os.path.join(dst_path, A[idx]))

def generate(args):
    GA = get_generator((512, 512, 3))
    GA.load_weights(os.path.join(args.cp_src, "GA_weights.h5"))
    GB = get_generator((512, 512, 3))
    GB.load_weights(os.path.join(args.cp_src, "GB_weights.h5"))
    
    process(GA, args.B_src, args.B2A_dst)
    process(GB, args.A_src, args.A2B_dst)
    
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--new', action = 'store_true', dest = 'train_new', default = False)

    parser.add_argument('-cs', '--checkpoint-source', dest = 'cp_src', type = str, default = 'models/checkpoint')
    parser.add_argument('-A', '--A-source', dest = 'A_src', type = str, default = 'res/A')
    parser.add_argument('-B', '--B-source', dest = 'B_src', type = str, default = 'res/B')
    parser.add_argument('-A2B', '--A2B-dst', dest = 'A2B_dst', type = str, default = 'res/A2B')
    parser.add_argument('-B2A', '--B2A-dst', dest = 'B2A_dst', type = str, default = 'res/B2A')

    args = parser.parse_args()

    generate(args)
