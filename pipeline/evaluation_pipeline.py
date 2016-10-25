import sys
import tensorflow as tf
import scipy.misc
import cv2
import utils
sys.path.insert(0, '../model/')
import driving_data

def main():
	final_mse = []
	model_configs = utils.from_recipe()
	for config in model_configs:
		NUM_ITER = config["NUM_ITER"]
		BATCH_SIZE = config["BATCH_SIZE"]
		MODEL_TITLE = config["MODEL_TITLE"]

		sess = tf.InteractiveSession()
		saver = tf.train.Saver()
		saver.restore(sess, "../save/model/{0}.ckpt".format(MODEL_TITLE))

		xs, ys_ = driving_data.test_xs, driving_data.test_ys

		with tf.name_scope('loss'):
    		loss = tf.reduce_mean(tf.square(tf.sub(y_, y)))
    		tf.scalar_summary('mse', loss)

	    mse = sess.run(loss, feed_dict={x: xs, y_: ys, keep_prob: 0.8})

	 	json_data = {"model": MODEL_TITLE, "mse": mse}
	 	final_mse.append(json_data)
	utils.to_json_file(final_mse, "report", "final_mse.json")
	best_model, best_mse = utils.find_best_model()
	print "Best Model is {0} With MSE {1}".format(best_model, best_mse)

if __name__ == '__main__':
	main()