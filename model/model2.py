__author__ = 'DanielMinsuKim'

import tensorflow as tf


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


class model(object):
    def __init__(self, data, WIDTH, HEIGHT, channel, n_class):
        self.x = tf.placeholder(tf.float32, [None, WIDTH * HEIGHT * channel])
        self.y_ = tf.placeholder(tf.float32, [None, n_class])
        self.y = self.model(WIDTH, HEIGHT, channel, n_class)
        self.data = data
    # Weight Initialization



    def model(self, WIDTH, HEIGHT, channel, n_class):
        x_image = tf.reshape(self.x, [-1, WIDTH, HEIGHT, channel])

        with tf.name_scope('input_reshape'):
            tf.image_summary('input', x_image, 10)

        ## first
        W_conv1 = weight_variable([5, 5, channel, 32])
        b_conv1 = bias_variable([32])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        ## second
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        ## third
        W_conv3 = weight_variable([5, 5, 64, 128])
        b_conv3 = bias_variable([128])

        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)

        h_pool2 = max_pool_2x2(h_conv3)

        ## forth

        W_conv4 = weight_variable([5, 5, 128, 256])
        b_conv4 = bias_variable([256])

        h_conv4 = tf.nn.relu(conv2d(h_pool2, W_conv4) + b_conv4)

        h_pool3 = max_pool_2x2(h_conv4)

        W_fc1 = weight_variable([(HEIGHT // 8) * (WIDTH // 8) * 256, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool3, [-1, (HEIGHT // 8) * (WIDTH // 8) * 256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # keep_prob = tf.placeholder(tf.float32)
        # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, n_class])
        b_fc2 = bias_variable([n_class])
        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

        return y_conv

    def feed_dict(self):
        batch = self.data.train.next_batch(100)
        return {self.x: batch[0], self.y_: batch[1]}