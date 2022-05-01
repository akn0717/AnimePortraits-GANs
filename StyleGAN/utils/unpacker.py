#This module is to unpack the h5 file into image files
import argparse
import io
import os

import cv2
import numpy as np
from alive_progress import alive_bar
import h5py
from PIL import Image

def unpacker(args):
    src_h5 = args.src
    target_dir = args.dst
    os.makedirs(target_dir, exist_ok = True)
    h5_file = h5py.File(src_h5, "r")
    print("Unpacking dataset...")
    with alive_bar(len(h5_file.keys())) as bar:
        for key in h5_file.keys():
            path = os.path.join(target_dir, str(key))
            img = np.array(h5_file[key])
            img = np.array(Image.open(io.BytesIO(img)))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, img)
            bar()
    print("Done!")
    h5_file.close()
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', dest = 'src', default = "dataset.h5")
    parser.add_argument('-d', dest = 'dst', default = "dataset")
    args = parser.parse_args()
    unpacker(args)