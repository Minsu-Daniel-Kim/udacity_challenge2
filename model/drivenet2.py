import tensorflow as tf
import scipy


class DriveNet2:

    def __init__(self, width, height, channel):

        self.height = height
        self.width = width
        self.channel = channel
        self.x = None
        self.y = None
        self.y_ = None
        self.num_pool = 0

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def inference(self):

        self.x = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channel])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 1])

        with tf.name_scope('input_reshape'):
            tf.image_summary('input', self.x, 10)

        # 1st convolutional layer
        with tf.name_scope('conv1') as scope:
            W_conv1 = self.weight_variable([5, 5, self.channel, 24])
            b_conv1 = self.bias_variable([24])
            h_conv1 = tf.nn.relu(self.conv2d(self.x, W_conv1, 2) + b_conv1)
            tf.histogram_summary("W_conv1", W_conv1);
            tf.histogram_summary("b_conv1", b_conv1);

        # 2nd convolutional layer
        with tf.name_scope('conv2') as scope:
            W_conv2 = self.weight_variable([5, 5, 24, 36])
            b_conv2 = self.bias_variable([36])
            h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2, 2) + b_conv2)
            tf.histogram_summary("W_conv2", W_conv2);
            tf.histogram_summary("b_conv2", b_conv2);

        with tf.name_scope('pool1') as scope:
            h_pool1 = self.max_pool_2x2(h_conv2)
            self.num_pool += 1

        # 3rd convolutional layer
        with tf.name_scope('conv3') as scope:
            W_conv3 = self.weight_variable([5, 5, 36, 48])
            b_conv3 = self.bias_variable([48])
            h_conv3 = tf.nn.relu(self.conv2d(h_pool1, W_conv3, 2) + b_conv3)
            tf.histogram_summary("W_conv3", W_conv3);
            tf.histogram_summary("b_conv3", b_conv3);

        # 4th convolutional layer
        with tf.name_scope('conv4') as scope:
            W_conv4 = self.weight_variable([3, 3, 48, 64])
            b_conv4 = self.bias_variable([64])
            h_conv4 = tf.nn.relu(self.conv2d(h_conv3, W_conv4, 1) + b_conv4)
            tf.histogram_summary("W_conv4", W_conv4);
            tf.histogram_summary("b_conv4", b_conv4);

        # 5th convolutional layer
        with tf.name_scope('conv5') as scope:
            W_conv5 = self.weight_variable([3, 3, 64, 64])
            b_conv5 = self.bias_variable([64])
            h_conv5 = tf.nn.relu(self.conv2d(h_conv4, W_conv5, 1) + b_conv5)
            tf.histogram_summary("W_conv5", W_conv5);
            tf.histogram_summary("b_conv5", b_conv5);

        with tf.name_scope('pool2') as scope:
            h_pool2 = self.max_pool_2x2(h_conv5)
            self.num_pool += 1

        # 6th convolutional layer
        with tf.name_scope('conv6') as scope:
            W_conv6 = self.weight_variable([3, 3, 64, 84])
            b_conv6 = self.bias_variable([84])
            h_conv6 = tf.nn.relu(self.conv2d(h_pool2, W_conv6, 1) + b_conv6)
            tf.histogram_summary("W_conv6", W_conv6);
            tf.histogram_summary("b_conv6", b_conv6);

        # 7th convolutional layer
        with tf.name_scope('conv6') as scope:
            W_conv7 = self.weight_variable([3, 3, 84, 110])
            b_conv7 = self.bias_variable([110])
            h_conv7 = tf.nn.relu(self.conv2d(h_conv6, W_conv7, 1) + b_conv7)
            tf.histogram_summary("W_conv7", W_conv7);
            tf.histogram_summary("b_conv7", b_conv7);

        with tf.name_scope('pool3') as scope:
            h_pool3 = self.max_pool_2x2(h_conv7)
            self.num_pool += 1

        # 1st fully connected layer
        with tf.name_scope('full1') as scope:
            params = (self.height // (2 ** self.num_pool)) * (self.width // (2 ** self.num_pool)) * 110
            W_fc1 = self.weight_variable([params, 1164])
            b_fc1 = self.bias_variable([1164])

            h_conv5_flat = tf.reshape(h_pool3, [-1, params])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)
            tf.histogram_summary("W_fc1", W_fc1);
            tf.histogram_summary("b_fc1", b_fc1);

        with tf.name_scope('dropout'):
            self.keep_prob = tf.placeholder(tf.float32)
            tf.scalar_summary('dropout_keep_probability', self.keep_prob)
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # 2nd fully connected layer
        with tf.name_scope('full2') as scope:
            W_fc2 = self.weight_variable([1164, 100])
            b_fc2 = self.bias_variable([100])

            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

            h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)
            tf.histogram_summary("W_fc2", W_fc2);
            tf.histogram_summary("b_fc2", b_fc2);

        # 3rd fully connected layer
        with tf.name_scope('full3') as scope:
            W_fc3 = self.weight_variable([100, 50])
            b_fc3 = self.bias_variable([50])

            h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

            h_fc3_drop = tf.nn.dropout(h_fc3, self.keep_prob)

        # 4th fully connected layer
        with tf.name_scope('full4') as scope:
            W_fc4 = self.weight_variable([50, 10])
            b_fc4 = self.bias_variable([10])

            h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)

            h_fc4_drop = tf.nn.dropout(h_fc4, self.keep_prob)

        # 5th fully connected layer
        with tf.name_scope('full5') as scope:
            W_fc5 = self.weight_variable([10, 1])
            b_fc5 = self.bias_variable([1])

        # output
        with tf.name_scope('output') as scope:
            self.y = tf.mul(tf.atan(tf.matmul(h_fc4_drop, W_fc5) + b_fc5), 2)
            tf.histogram_summary("output", self.y);