import os.path
import json
import numpy as np
import pickle


# Serializes DATA under NAME
# Serialize to save/DIRECTORY/name
# Remember to add extension manually if there needs to be any.
def to_json_file(data, directory, name):
    path_name = '../save/' + directory + '/' + name
    with open(path_name, 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))


# Opens up a file located in save/DIRECTORY/FILE_NAME
def from_json_file(directory, file_name):
    path_name = '../save/' + directory + '/' + file_name
    with open(path_name) as data_file:    
        data = json.load(data_file)
    return data

# Given a list of models to deserialize, returns a list with all the models
# Assume that recipe contains list of strings
def from_recipe():
    all_the_models = []
    with open('../save/config/recipe.json') as recipe:
        to_deserialize = json.load(recipe)
    for ingredient in to_deserialize:
        all_the_models += [from_json_file("config", ingredient)]
    return all_the_models

# Appends the ENTRY = {iter, mse} to the model specified
def append(desired_model, entry):
    if not os.path.isfile("../save/report/report.json"):
        all_models = []
        all_models.append({"model":desired_model, "mse":[entry]})
    else:
        model_exists = False
        all_models = from_json_file("report", "report.json")
        for model in all_models:
            if model["model"] == desired_model:
                model_exists = True
                model["mse"] += [entry]
                break
        if not model_exists:
            all_models.append({"model":desired_model, "mse":[entry]})
    to_json_file(all_models, "report", "report.json")


# Reads report.json, and finds the best model
# Returns a string (name of best model)
def find_best_model():
    reports = from_json_file("report", "final_mse.json")
    lowest_mse = float('inf')
    best_model = None
    for model in reports:
        if model["mse"] < lowest_mse:
            lowest_mse = model["mse"]
            best_model = model["model"]
    return best_model, lowest_mse
    # best_models = [[model1, min_mse], [model2, min_mse]]
    # best_models = []
    # for model in reports:
    # 	best_models += [[model["model"], model["mse"].pop()["mse"]]]
    # best_model_name = best_models[0][0]
    # best_model_score = best_models[0][1]
    # for candidate in best_models:
    # 	if candidate[1] < best_model_score:
    # 		best_model_name = candidate[0]
    # 		best_model_score = candidate[1]
    # return best_model_name

def normalizeImage(matrix):
    pass

def gather_flattened_image(matrix, data_gathered):
    flattened = flatten_matrix(matrix)
    if data_gathered == None:
        data_gathered = flattened
    else:
        data_gathered = np.vstack((data_gathered, flattened))
    return data_gathered

def flatten_matrix(matrix):
    vector = matrix.flatten(1)
    vector = vector.reshape(1, len(vector))
    return vector


# def zca_whitening(inputs):
#     sigma = np.dot(inputs, inputs.T)/inputs.shape[1] #Correlation matrix
#     U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
#     epsilon = 0.1                #Whitening constant, it prevents division by zero
#     ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T) #ZCA Whitening matrix
#     with open('../save/pickle/zca_matrix.pickle', 'wb') as outfile:
#         pickle.dump(ZCAMatrix, outfile)

#     # pickle.load('../save/pickle/zca_matrix.pickle')
#     return np.dot(ZCAMatrix, inputs)   #Data whitening

def build_recipe_and_model_configs():
    model_configs = []

    model_configs.append({"MODEL_TITLE":"ALEXNET", \
                   "MODEL_FILE":"alexnet.json", \
                   "NUM_ITER":1000, \
                   "BATCH_SIZE":100, \
                   "WIDTH":100, \
                   "HEIGHT":100, \
                   "CHANNEL":3})

    model_configs.append({"MODEL_TITLE":"LENET", \
                   "MODEL_FILE":"lenet.json", \
                   "NUM_ITER":1000, \
                   "BATCH_SIZE":100, \
                   "WIDTH":100, \
                   "HEIGHT":100, \
                   "CHANNEL":3})

    model_configs.append({"MODEL_TITLE":"NVIDIANET", \
                   "MODEL_FILE":"nvidianet.json", \
                   "NUM_ITER":1000, \
                   "BATCH_SIZE":100, \
                   "WIDTH":100, \
                   "HEIGHT":100, \
                   "CHANNEL":3})

    recipe = [config["MODEL_FILE"] for config in model_configs]
    to_json_file(recipe, "config", "recipe.json")

    for config in model_configs:
        to_json_file(config, "config", config["MODEL_FILE"])

build_recipe_and_model_configs()