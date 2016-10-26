import scipy.misc
import random
import numpy as np
from sklearn.model_selection import train_test_split

import json

class Dataset:

    def __init__(self, DATA_DIR, file_name, width, height):

        self.WIDTH = width
        self.HEIGHT = height
        self.train_batch_pointer = 0
        self.val_batch_pointer = 0
        self.DATA_DIR = DATA_DIR
        self.file_name = file_name
        self.num_images = 0
        self.train_xs = None
        self.train_ys = None
        self.val_xs = None
        self.val_ys = None
        self.test_xs = None
        self.test_ys = None
        self.num_train_images = 0
        self.num_val_images = 0
        self.num_test_images = 0
        # initialize dataset
        self.prepare_data()

    def prepare_data(self):
        xs = []
        ys = []

        with open(self.DATA_DIR + '/' + self.file_name) as f:
            for line in f:
                xs.append(self.DATA_DIR + "/" + line.split()[0])
                ys.append(float(line.split()[1]) * scipy.pi / 180)
        self.num_images = len(xs)
        c = list(zip(xs, ys))
        random.shuffle(c)
        xs, ys = zip(*c)

        self.train_xs, self.test_xs, self.train_ys, self.test_ys \
            = train_test_split(xs, ys, test_size=0.20, random_state=42)
        self.train_xs, self.val_xs, self.train_ys, self.val_ys \
            = train_test_split(self.train_xs, self.train_ys, test_size=0.20, random_state=42)
        self.num_train_images = len(self.train_xs)
        self.num_val_images = len(self.val_xs)
        self.num_test_images = len(self.test_xs)

    def LoadTrainBatch(self, batch_size):
        # global train_batch_pointer
        x_out = []
        y_out = []
        for i in range(0, batch_size):
            x_out.append(scipy.misc.imresize(
                scipy.misc.imread(self.train_xs[(self.train_batch_pointer + i) % self.num_train_images])[-150:], [self.HEIGHT, self.WIDTH]) / 255.0)
            y_out.append([self.train_ys[(self.train_batch_pointer + i) % self.num_train_images]])
        self.train_batch_pointer += batch_size
        return x_out, y_out

    def LoadValBatch(self, batch_size):
        # global self.val_batch_pointer
        x_out = []
        y_out = []
        for i in range(0, batch_size):
            x_out.append(scipy.misc.imresize(scipy.misc.imread(self.val_xs[(self.val_batch_pointer + i) % self.num_val_images])[-150:], [self.HEIGHT, self.WIDTH]) / 255.0)
            y_out.append([self.val_ys[(self.val_batch_pointer + i) % self.num_val_images]])
        self.val_batch_pointer += batch_size
        return x_out, y_out
