import os
import json

def saveReport(models, values):
	jsonData = list()
	labels = ["a","b","c","d"]
	for i in range(len(models)):
		subData = dict()
		subData["_model"] = models[i]
		value = values[i]
		for j in range(len(value)):
			subData[labels[j]] = value[j]
		jsonData.append(subData)
	with open('report.json', 'w') as outfile:
	     json.dump(jsonData, outfile, indent = 4, ensure_ascii=False)

def saveModel(modelName):
	saver = tf.train.Saver()
	with tf.Session() as sess:
		save_path = saver.save(sess, "/tmp/{0}.ckpt".format(modelName))
		print("Model saved in file: %s" % save_path)

def saveConfig(models, params):
	labels = ["a","b","c","d"]
	for i in range(len(models)):
		jsonData = dict()
		model = models[i]
		param = params[i]
		for j in range(len(param)):
			jsonData[labels[j]] = param[j]
		with open("model_%s.json" % model, 'w') as outfile:
			json.dump(jsonData, outfile, indent = 4, ensure_ascii=False)

def output(models, values, params):
	saveConfig(models, params)
	saveReport(models, values)

if __name__ == '__main__':
	# output(...)
  saveReport(["alexNet", "leNet"], [['1','2','3','4'],['7','2','4','5']])