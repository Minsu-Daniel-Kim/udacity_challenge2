import os
import json


# Serializes DATA under NAME
# Serialize to save/DIRECTORY/name
# Remember to add extension manually if there needs to be any.
def to_json_file(data, directory, name):
    path_name = '../save/' + directory + '/' + name
    with open(path_name, 'w') as f:
        json.dump(data, f)


# Opens up a file located in save/DIRECTORY/FILE_NAME
def from_json_file(directory, file_name):
    path_name = '../save/' + directory + '/' + name
    with open(path_name) as data_file:    
        data = json.load(data_file)
    print (data)
    return data

# Given a list of models to deserialize, returns a list with all the models
# Assume that recipe contains list of strings
def from_recipe():
    all_the_models = []
    with open('../save/config/recipe.json') as recipe:
        to_deserialize = json.load(recipe)
    for ingredient in to_deserialize:
    	all_the_models += [from_json_file(config, ingredient)]
    return all_the_models

   

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
	with open('../save/report.json', 'w') as outfile:
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