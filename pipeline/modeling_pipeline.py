import os
import tensorflow as tf
import driving_data
import model
from drivenet import DriveNet
import utils

def train():

    models = utils.from_recipe()
    for model in models:
        config = utils.from_json_file("config", "%s.ckpt" % model)
        # config values
        NUM_ITER = config["NUM_ITER"]
        BATCH_SIZE = config["BATCH_SIZE"]
        MODEL_TITLE = config["MODEL_TITLE"]
        LOGDIR = '../save/model'
        summaries_dir = '/tmp/' + MODEL_TITLE

        sess = tf.InteractiveSession()

        model = DriveNet(width=config["WIDTH"], height=config["HEIGHT"], channel=config["CHANNEL"])
        y, y_, x, keep_prob = model.inference()

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.square(tf.sub(y_, y)))
            tf.scalar_summary('mse', loss)

        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        sess.run(tf.initialize_all_variables())

        merged = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(summaries_dir + '/train',
                                              sess.graph)
        test_writer = tf.train.SummaryWriter(summaries_dir + '/test')

        saver = tf.train.Saver()

        for i in range(NUM_ITER):
            xs, ys = driving_data.LoadTrainBatch(BATCH_SIZE)
            if i % 10 == 0:
                summary, acc = sess.run([merged, loss], feed_dict={x: xs, y_: ys, keep_prob: 0.8})
                test_writer.add_summary(summary, i)
                print('MSE at step %s: %s' % (i, acc))
            else:  # Record train set summaries, and train

                if i % 100 == 99:  # Record execution stats
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged, train_step],
                                          feed_dict={x: xs, y_: ys, keep_prob: 0.8},
                                          options=run_options,
                                          run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                    train_writer.add_summary(summary, i)
                    print('Adding run metadata for', i)
                else:  # Record a summary
                    summary, _ = sess.run([merged, train_step], feed_dict={x: xs, y_: ys, keep_prob: 0.8})
                    train_writer.add_summary(summary, i)
            if i % 100 == 0:
                if not os.path.exists(LOGDIR):
                    os.makedirs(LOGDIR)

                checkpoint_path = os.path.join(LOGDIR, MODEL_TITLE + ".ckpt")
                filename = saver.save(sess, checkpoint_path)
                print("Model saved in file: %s" % filename)

                json_data = {"iter":i, "acc":acc}
                utils.append(model, json_data)
                print("Model saved in file: %s" % filename)

        train_writer.close()
        test_writer.close()
     
def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
