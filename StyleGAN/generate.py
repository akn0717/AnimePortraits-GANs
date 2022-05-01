import argparse
import os


def generate(args):
    pass

def catch_exceptions(args):
    checkpoint_exist = (os.path.isfile("models/checkpoint/generator_weights.h5")
                        or os.path.isfile("models/checkpoint/mapping_weights.h5")
                        or os.path.isfile("models/checkpoint/log.json")
                        or os.path.isfile("models/checkpoint/model.h5"))

    return not(checkpoint_exist)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', dest = 'batch_size', type = int, default = 4)
    parser.add_argument('-cs', '--checkpoint-src', dest = 'checkpoint_source', type = str, default = 'models/checkpoint')

    parser.add_argument('-m', '--mode', dest = 'mode', type = int, default = 0)
    parser.add_argument('-b1', '--beta_1', dest = 'beta_1', type = float, default = 1.0)
    parser.add_argument('-b2', '--beta_2', dest = 'beta_2', type = float, default = 1.0)
    args = parser.parse_args()

    if not(catch_exceptions(args)):
        generate(args)
