import tensorflow as tf
import scipy.misc
import model
import cv2
import driving_data
from subprocess import call

def main():
	models = utils.from_recipe()
	for model in models:
		config = utils.from_json_file("config", "%s.ckpt" % model)
    # config values
    NUM_ITER = config["NUM_ITER"]
    BATCH_SIZE = config["BATCH_SIZE"]
    MODEL_TITLE = config["MODEL_TITLE"]
    # LOGDIR = '../save/model'
    # summaries_dir = '/tmp/' + MODEL_TITLE

		sess = tf.InteractiveSession()
		saver = tf.train.Saver()
		saver.restore(sess, "save/{0}.ckpt".format(model))

		xs, ys_ = driving_data.val_xs, driving_data.val_ys


		# img = cv2.imread('steering_wheel_image.jpg',0)
		rows,cols = img.shape

		smoothed_angle = 0

		cap = cv2.VideoCapture(0)
		while(cv2.waitKey(10) != ord('q')):
		    ret, frame = cap.read()
		    image = scipy.misc.imresize(frame, [66, 200]) / 255.0
		    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180 / scipy.pi