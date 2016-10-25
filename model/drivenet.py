import tensorflow as tf
import scipy


class DriveNet:

    def __init__(self, width, height, channel):

        self.height = height
        self.width = width
        self.channel = channel
        self.x = None
        self.y = None
        self.y_ = None

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

    def inference(self):

        self.x = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channel])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 1])

        # 1st convolutional layer
        W_conv1 = self.weight_variable([5, 5, self.channel, 24])
        b_conv1 = self.bias_variable([24])

        h_conv1 = tf.nn.relu(self.conv2d(self.x, W_conv1, 2) + b_conv1)

        # 2nd convolutional layer
        W_conv2 = self.weight_variable([5, 5, 24, 36])
        b_conv2 = self.bias_variable([36])

        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2, 2) + b_conv2)

        # 3rd convolutional layer
        W_conv3 = self.weight_variable([5, 5, 36, 48])
        b_conv3 = self.bias_variable([48])

        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 2) + b_conv3)

        # 4th convolutional layer
        W_conv4 = self.weight_variable([3, 3, 48, 64])
        b_conv4 = self.bias_variable([64])

        h_conv4 = tf.nn.relu(self.conv2d(h_conv3, W_conv4, 1) + b_conv4)

        # 5th convolutional layer
        W_conv5 = self.weight_variable([3, 3, 64, 64])
        b_conv5 = self.bias_variable([64])

        h_conv5 = tf.nn.relu(self.conv2d(h_conv4, W_conv5, 1) + b_conv5)

        # 1st fully connected layer
        W_fc1 = self.weight_variable([1152, 1164])
        b_fc1 = self.bias_variable([1164])

        h_conv5_flat = tf.reshape(h_conv5, [-1, 1152])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)

        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # 2nd fully connected layer
        W_fc2 = self.weight_variable([1164, 100])
        b_fc2 = self.bias_variable([100])

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)

        # 3rd fully connected layer
        W_fc3 = self.weight_variable([100, 50])
        b_fc3 = self.bias_variable([50])

        h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

        h_fc3_drop = tf.nn.dropout(h_fc3, self.keep_prob)

        # 4th fully connected layer
        W_fc4 = self.weight_variable([50, 10])
        b_fc4 = self.bias_variable([10])

        h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)

        h_fc4_drop = tf.nn.dropout(h_fc4, self.keep_prob)

        # 5th fully connected layer
        W_fc5 = self.weight_variable([10, 1])
        b_fc5 = self.bias_variable([1])

        # output
        self.y = tf.mul(tf.atan(tf.matmul(h_fc4_drop, W_fc5) + b_fc5), 2)
