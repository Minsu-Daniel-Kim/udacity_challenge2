def preprocessImage(imagePath, csvPath):
	csvFile = open(csvPath, "r")
	angles = []
	for line in csvFile:
		_, _, angle, _, _ = line.split(",")
		angles.append(float(angle))
	readableImages = [ f for f in listdir(imagePath) if isfile(join(imagePath,f))]

	jsonData = []
	for i in range(0, len(readableImages)):
		jsonDict = {"image":readableImages[i], "angle":angles[i]}
		jsonData.append(jsonDict)
	with open('data.json', 'w') as outfile:
		json.dump(jsonData, outfile, indent = 4, ensure_ascii=False)

# if __name__ == '__main__':
		# preprocessImage()