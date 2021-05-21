import h5py

import numpy as np
import argparse
import os
from utils.imglib import *


def customize(args):
    custom_data_dir = args.source

    H = W = 64
    if (int(args.size) == 1):
        H = W = 256

    img_files = [f for f in os.listdir(custom_data_dir) if os.path.isfile(os.path.join(custom_data_dir, f))]
    cnt = 0
    with h5py.File(args.dest, mode = 'w') as f:
        for file in img_files:
            cnt+=1
            if (cnt%1000)==0:
                print(cnt,'/',len(img_files))
            img = load_image(os.path.join(custom_data_dir,file),H,W)
            f.create_dataset(file, data = img, compression = "gzip", chunks = True)

    return 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', dest = 'source', default = 'custom_dataset')
    parser.add_argument('-r', '--resolution', dest = 'size', default = 0)
    parser.add_argument('-d', '--destination', dest = 'dest', default = 'dataset.h5')
    args = parser.parse_args()
    
    customize(args)
