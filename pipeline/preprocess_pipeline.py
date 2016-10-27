from os import listdir
from os.path import isfile, join
import argparse
import numpy as np
import cv2
import json
import utils
import pylab

""" 
Given Paths, reads images in paths and does all preprocesses marked in ARGS.
All processed images are stored in new directories with appended suffix "_processed".
"""
def preprocessImage(imagePath):
	processedImagePath = imagePath + "_processed"
	readableImages = [ f for f in listdir(imagePath) if not f.startswith('.') and isfile(join(imagePath,f)) ]
	
	m,n,l = 256, 455, 3
	flat_len = m*n*l
	imnbr = len(readableImages)
	immatrix = np.empty((0, flat_len), float)

	for i in range(imnbr):
		image = cv2.imread(join(imagePath,readableImages[i]))
		image = image.reshape([1, flat_len])
		if args.normalized:
			image = utils.normalizeImage(image)
		if args.zca:
			immatrix = np.vstack((immatrix, image))

	x_zca = utils.zca(immatrix)
	# TODO pickle immatrix, x_zca

	for i in range(imnbr):
		imageMat = x_zca[i,:]
		print (imageMat.shape)
		imageProcessed = imageMat.reshape(m, n, l)
		cv2.imwrite(join(processedImagePath, readableImages[i]), imageProcessed)

def readConfig(filePath):
	with open(filePath) as config_file:
		config = json.load(config_file)
		argDict = vars(args)
		for k, v in config.items():
			if k in argDict.keys():
				argDict[k] = eval(v)
			else:
				print ("ERROR: Given Option Does Not Exist.")
				sys.exit()

def main():
	parser = argparse.ArgumentParser(description='Preprocess Image files and Store processed Images.')
	parser.add_argument('-z', '--zca', type=bool, nargs='?', default=False, help='ZCA parameter')
	parser.add_argument('-n', '--normalized', type=bool, nargs='?', default=False, help='normalized parameter')
	parser.add_argument('-c', '--config', type=str, nargs='?', default='', help='Config file')
	parser.set_defaults(debug=False)
	global args
	args = parser.parse_args()
	if len(args.config) != 0:
		readConfig(args.config)
	preprocessImage("../rawdata/driving_dataset")

if __name__ == '__main__':
	main()
