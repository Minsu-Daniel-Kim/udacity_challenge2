from os import listdir
from os.path import isfile, join
import sys
import numpy
import cv2
import json

argDict = {"zca": False, "normalized": False}

def preprocessImage(path):
	imagePath="/Volumes/DANIEL/dataset/{0}".format(path)
	readableImages = [ f for f in listdir(imagePath) if isfile(join(imagePath,f)) ]
	print len(readableImages)
	images = numpy.empty(len(readableImages), dtype=object)
	for n in range(0, len(readableImages)):
	  image = cv2.imread( join(imagePath,readableImages[n]) )
	  print type(image)
	  # if argDict["zca"]:
	  # 	image = zcaImage(image[n])
	  # if argDict["normalized"]:
	  # 	image = normalizeImage(zcaImage)
	  # images[n] = image

	# save images
	return images

def readConfig(filePath):
	with open(filePath) as config_file:
		config = json.load(config_file)
		for k, v in config.items():
			if k in argDict.keys():
				argDict[k] = v
			else:
				print "ERROR: Given Option Does Not Exist."
				sys.exit()

def main(args):
		# preprocessImage("center")
	for k, v in args.items():
		print k, v

if __name__ == '__main__':
	print "\nUSAGE: python preprocess_pipeline.py [--option]=true/false"
	print "USAGE: python preprocess_pipeline.py [PATH_TO_CONFIG]\n"
	args = sys.argv
	if len(args) == 2 and '=' not in args[1]:
		readConfig(args[1])
	else:
		for i in range(len(args)-1):
			k, v = args[i+1].split('=')
			k = k.lstrip().rstrip()[2:]
			v = v.lstrip().rstrip()
			if k not in argDict.keys():
				print "ERROR: Given Option Does Not Exist."
				sys.exit()
			else:
				argDict[k] = eval(v.title())
	main(argDict)
