import os
from os import listdir
from os.path import isfile, join
import argparse
import numpy as np
import cv2
import json
import pickle

SAVE_DIR = "save"
PICKLE_DIR = "pickle"

def flattenImage(readableImages):
	"""
	Flattens readableImages in IMAGEPATH and stacks 1-d vectors in flat_image_data.
	flat_image_data is stored as a serialized object for future use.

	Input:
	readableImages -- List of images to be flattened.
	"""
	imagePath = args.imagePath
	m,n,l = cv2.imread(join(imagePath, readableImages[0])).shape
	partition = args.partition
	low = args.totalImageNum * partition // args.totalPartitionNum
	high = args.totalImageNum * (partition+1) // args.totalPartitionNum
	print ("Flattening %dth~%dth image..." % (low, high-1))

	flat_image_data = []
	j = 0
	for i in range(low, high):
		flat_image_datum = dict()
		if args.verbose and (j % 100 == 0 and j != 0):
			print ("Partition %d: Processed %2.1f%% (%d/%d) images" % (partition, j*100/(high-low), j, (high-low)))
		image = cv2.imread(join(imagePath,readableImages[i]))
		image = image.reshape([1, m*n*l])
		flat_image_datum["id"] = readableImages[i]
		flat_image_datum["mat"] = image
		flat_image_data.append(flat_image_datum)
		j+=1
	if args.verbose:
		print ("Partition %d: Saving Serialized flat_image_data..." % partition)
	save_pickle(flat_image_data, PICKLE_DIR, "flat_image_data_%d.pickle" % partition)

	if args.verbose:
		print ("Flattening %dth Partition Finished" % partition)

def preprocessImage():
	""" 
	Reads images in IMAGEPATH, and does all preprocesses marked in ARGS.
	All processed images are stored in processedImagePath.
	"""
	imagePath = args.imagePath
	dstPath = args.dstPath
	if dstPath == '':
		dstPath = imagePath + "_processed"
		if not os.path.exists(dstPath):
			os.makedirs(dstPath)
	readableImages = [f for f in listdir(imagePath) \
			if not f.startswith('.') and isfile(join(imagePath,f)) and f.endswith('.jpg')]
	if args.totalImageNum == -1:
		args.totalImageNum = len(readableImages)
	
	if not args.useStoredPickle:
		flattenImage(readableImages)

# Saves given DATA into SAVE_DIR/DIRECTORY/NAME
def save_pickle(data, directory, name):
    path_name = SAVE_DIR + '/' + directory + '/' + name
    with open(path_name, 'wb') as f:
        pickle.dump(data, f)

# Reads serialized data stored in SAVE_DIR/DIRECTORY/NAME
# RETURNS loaded serialized data, DATA.
def load_pickle(directory, name):
    path_name = SAVE_DIR + '/' + directory + '/' + name
    with open(path_name) as f:
        data = pickle.load(f)
    return data

def main():
	parser = argparse.ArgumentParser(description='Preprocess Image files and Store processed Images.')
	parser.add_argument('-z', '--zca', type=bool, nargs='?', default=False, help='ZCA parameter')
	parser.add_argument('-n', '--normalized', type=bool, nargs='?', default=False, help='normalized parameter')
	parser.add_argument('-useStoredPickle', '--useStoredPickle', type=bool, nargs='?', default=False, help='Use Preprocessed pickle')
	parser.add_argument('-verbose', '--verbose', type=bool, nargs='?', default=False, help='Verbose')
	parser.add_argument('-partition', '--partition', type=int, nargs='?', default=0, help='ith partition')
	parser.add_argument('-totalPartitionNum', '--totalPartitionNum', type=int, nargs='?', default=3, help='Total num of image set partition')
	parser.add_argument('-totalImageNum', '--totalImageNum', type=int, nargs='?', default=-1, help='Total num of images')
	parser.add_argument('-imagePath', '--imagePath', type=str, nargs='?', default="rawdata/driving_dataset", help='Image Files Path')
	parser.add_argument('-dstPath', '--dstPath', type=str, nargs='?', default='', help='Processed Image Files Path')
	parser.set_defaults(debug=False)
	global args
	args = parser.parse_args()
	preprocessImage()

if __name__ == '__main__':
	main()
