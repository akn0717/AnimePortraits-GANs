import numpy as np
from StyleGAN_model import *

def interpolate_points(v1, v2, n_steps = 10):
    ratios = np.linspace(0,1,num = n_steps)

    v = []
    for ratio in ratios:
        v.append((1-ratio)*v1 + ratio*v2)

    return v

if __name__ == "__main__":
    img_weight = img_height = 256
    channels = 3
    model = StyleGAN(1, 256, 256, 3, load_model = True, load_const = True)

