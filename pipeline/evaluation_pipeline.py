import tensorflow as tf
import scipy.misc
import model
import cv2
import driving_data
from subprocess import call

def main():
	final_accuracy = []
	models = utils.from_recipe()
	for model in models:
		config = utils.from_json_file("config", "%s.ckpt" % model)
    # config values
    NUM_ITER = config["NUM_ITER"]
    BATCH_SIZE = config["BATCH_SIZE"]
    MODEL_TITLE = config["MODEL_TITLE"]

		sess = tf.InteractiveSession()
		saver = tf.train.Saver()
		saver.restore(sess, "../save/model/{0}.ckpt".format(MODEL_TITLE))

		xs, ys_ = driving_data.test_xs, driving_data.test_ys
		# ys = model.eval(xs)

		with tf.name_scope('loss'):
      loss = tf.reduce_mean(tf.square(tf.sub(y_, y)))
      tf.scalar_summary('mse', loss)

    mse = sess.run(loss, feed_dict={x: xs, y_: ys, keep_prob: 0.8})
    json_data = {"model": MODEL_TITLE, "mse": mse}
    final_accuracy.append(json_data)
  
  utils.to_json_file(final_accuracy, "report", "final_accuracy.json")

if __name__ == '__main__':
	main()