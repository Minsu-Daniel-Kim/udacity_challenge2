import os
import tensorflow as tf
import sys
sys.path.insert(0, '../model/')
from drivenet import DriveNet
import driving_data
import utils

from random import randint


LOGDIR = '../save/model'


def train():

    model_dict = {

        'driveNet': DriveNet

    }
    model_configs = utils.from_recipe()

    for config in model_configs:

        # config
        NUM_ITER = config["NUM_ITER"]
        BATCH_SIZE = config["BATCH_SIZE"]
        MODEL_TITLE = config["MODEL_TITLE"]
        MODEL_FILE = config["MODEL_FILE"]
        SUMMARY_DIR = '/tmp/' + MODEL_TITLE

        # get session
        sess = tf.InteractiveSession()


        # setup model
        dnn = model_dict[config['MODEL_TITLE']](width=config["WIDTH"], height=config["HEIGHT"], channel=config["CHANNEL"])
        dnn.inference()

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.square(tf.sub(dnn.y_, dnn.y)))
            tf.scalar_summary('mse', loss)


        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        sess.run(tf.initialize_all_variables())

        # summary statistics
        merged = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(SUMMARY_DIR + '/train', sess.graph)
        test_writer = tf.train.SummaryWriter(SUMMARY_DIR + '/test')

        # saver
        saver = tf.train.Saver()

        for i in range(NUM_ITER):
            xs, ys = driving_data.LoadTrainBatch(BATCH_SIZE)
            if i % 10 == 0:
                xs, ys = driving_data.LoadValBatch(100)
                summary, mse = sess.run([merged, loss], feed_dict={dnn.x: xs, dnn.y_: ys, dnn.keep_prob: 0.8})
                test_writer.add_summary(summary, i)
            elif i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict={dnn.x: xs, dnn.y_: ys, dnn.keep_prob: 0.8},
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:
                summary, _ = sess.run([merged, train_step], feed_dict={dnn.x: xs, dnn.y_: ys, dnn.keep_prob: 0.8})
                train_writer.add_summary(summary, i)

            if i % 100 == 0:
                if not os.path.exists(LOGDIR):
                    os.makedirs(LOGDIR)
                #
                checkpoint_path = os.path.join(LOGDIR, MODEL_TITLE + ".ckpt")
                filename = saver.save(sess, checkpoint_path)
                print("Model saved in file: %s" % filename)

                json_data = {"iter": i, "mse": mse}
                utils.append(MODEL_TITLE, json_data)

        train_writer.close()
        test_writer.close()
     
def main(_):
    train()

if __name__ == '__main__':
    tf.app.run()
