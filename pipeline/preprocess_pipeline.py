from os import listdir
from os.path import isfile, join
import argparse
import numpy
import cv2
import json
import utils

""" 
Given Paths, reads images in paths and does all preprocesses marked in ARGS.
All processed images are stored in new directories with appended suffix "_processed".
"""
def preprocessImage(imagePath):
	processedImagePath = imagePath + "_processed"
	readableImages = [ f for f in listdir(imagePath) if not f.startswith('.') and isfile(join(imagePath,f)) ]
	image_flattened = None
	for n in range(0, len(readableImages)):
		image = cv2.imread( join(imagePath,readableImages[n]) )
	#   # if args.zca:
	#   # 	image_flattened = utils.gather_flattened_image(image, image_flattened)
		if args.normalized:
			image = utils.normalizeImage(image)
		cv2.imwrite(join(processedImagePath, readableImages[n]), image)
	# trf = ZCA().fit(image_flattened)
	# image_whitened = trf.transform(image_flattened)

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
