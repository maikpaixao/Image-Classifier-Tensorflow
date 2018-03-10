import argparse

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Build a classifier using \
            MNIST data')
    parser.add_argument('--input-dir', dest='input_dir', type=str,
            default='./mnist_data', help='Directory for storing data')
    return parser