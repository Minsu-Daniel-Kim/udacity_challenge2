import os
import tensorflow as tf
import driving_data
import model
# import alexnet
# model = alexnet


# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
# flags.DEFINE_integer('n_bundle', 1, 'Number of bundles to upload.')
# flags.DEFINE_integer('validation_size', 2000, 'Number of bundles to upload.')
# flags.DEFINE_string('data_dir', '/tmp/data2', 'Directory for storing data')
# flags.DEFINE_string('summaries_dir', '/tmp/mnist_logs2', 'Summaries directory')
LOGDIR = './save'

summaries_dir = '/tmp/mnist_logs2'

def train():

    sess = tf.InteractiveSession()

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(tf.sub(model.y_, model.y)))
        tf.scalar_summary('mse', loss)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    sess.run(tf.initialize_all_variables())

    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(summaries_dir + '/train',
                                          sess.graph)
    test_writer = tf.train.SummaryWriter(summaries_dir + '/test')

    saver = tf.train.Saver()

    # train over the dataset about 30 times
    for i in range(100000):
        xs, ys = driving_data.LoadTrainBatch(100)
        if i % 10 == 0:
            summary, acc = sess.run([merged, loss], feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.8})
            test_writer.add_summary(summary, i)
            print('MSE at step %s: %s' % (i, acc))
        else:  # Record train set summaries, and train

            if i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.8},
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.8})
                train_writer.add_summary(summary, i)
        if i % 100 == 0:
            if not os.path.exists(LOGDIR):
                os.makedirs(LOGDIR)
            checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
            filename = saver.save(sess, checkpoint_path)
            print("Model saved in file: %s" % filename)

    train_writer.close()
    test_writer.close()
     
def main(_):

    train()


if __name__ == '__main__':
    tf.app.run()
