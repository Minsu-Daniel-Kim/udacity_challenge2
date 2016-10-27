from os import listdir
from os.path import isfile, join
import argparse
import numpy as np
import cv2
import json
import utils

def flattenImage(imagePath, readableImages):
	"""
	Flattens readableImages in IMAGEPATH and stacks 1-d vectors in immatrix.
	immatrix is stored as a serialized object for future use.

	Input:
	imagePath -- Image path for images.
	readableImages -- List of images to be flattened.

	Output:
	n-d array, which is an aggregation of all flattened image vectors.
	"""
	m,n,l = cv2.imread(join(imagePath, readableImages[0])).shape
	flatLen = m*n*l
	imnbr = len(readableImages)
	if args.pickle_path != '':
		immatrix = utils.load_pickle(args.pickle_path)
	else:
		immatrix = np.empty((0, flatLen), float)

		for i in range(imnbr):
			if args.verbose and (i % 100 == 0 and i != 0):
				print ("Processed %d images" % i)
			image = cv2.imread(join(imagePath,readableImages[i]))
			image = image.reshape([1, flatLen])
			immatrix = np.vstack((immatrix, image))

		# Save immatrix for future use.
		if args.verbose:
			print ("Save Serialized immatrix...")
		utils.save_pickle(immatrix, 'pickle', 'immatrix.pickle')
	return immatrix

def preprocessImage(imagePath, processedImagePath):
	""" 
	Reads images in IMAGEPATH, and does all preprocesses marked in ARGS.
	All processed images are stored in processedImagePath.

	Input:
	imagePath -- Image path for images.
	processedImagePath -- Path for processed image data.
	"""
	readableImages = [f for f in listdir(imagePath) \
			if not f.startswith('.') and isfile(join(imagePath,f))]
	imnbr = len(readableImages)
	immatrix = flattenImage(imagePath, readableImages)
	
	if args.normalized:
		if args.verbose:
			print ("Normalize images...")
		immatrix = utils.normalize(immatrix)

	if args.zca:
		if args.verbose:
			print ("ZCA images...")
		immatrix = utils.zca(immatrix)

	for i in range(imnbr):
		imageProcessed = immatrix[i,:].reshape(m, n, l)
		cv2.imwrite(join(processedImagePath, readableImages[i]), imageProcessed)

def main():
	parser = argparse.ArgumentParser(description='Preprocess Image files and Store processed Images.')
	parser.add_argument('-z', '--zca', type=bool, nargs='?', default=False, help='ZCA parameter')
	parser.add_argument('-n', '--normalized', type=bool, nargs='?', default=False, help='normalized parameter')
	parser.add_argument('-p', '--pickle_path', type=str, nargs='?', default='', help='Flattened imageMat pickle path')
	parser.add_argument('-v', '--verbose', type=bool, nargs='?', default=False, help='Verbose')
	parser.set_defaults(debug=False)
	global args
	args = parser.parse_args()
	if len(args.config) != 0:
		readConfig(args.config)
	preprocessImage("../rawdata/driving_dataset", "../rawdata/driving_dataset_processed")

if __name__ == '__main__':
	main()
