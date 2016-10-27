import os.path
import json
import numpy as np
import pickle
from scipy import linalg
# from sklearn.utils import array2d, as_float_array
# from sklearn.base import TransformerMixin, BaseEstimator
from sklearn import preprocessing
import matplotlib.pyplot as plt

SAVE_DIR = "../save"

# Serializes DATA under NAME
# Serialize to save/DIRECTORY/name
# Remember to add extension manually if there needs to be any.
def to_json_file(data, directory, name):
    path_name = SAVE_DIR + '/' + directory + '/' + name
    with open(path_name, 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))


# Opens up a file located in save/DIRECTORY/FILE_NAME
def from_json_file(directory, file_name):
    path_name = SAVE_DIR + '/' + directory + '/' + file_name
    with open(path_name) as data_file:
        data = json.load(data_file)
    return data


# Given a list of models to deserialize, returns a list with all the models
# Assume that recipe contains list of strings
def from_recipe():
    all_the_models = []
    with open(SAVE_DIR + '/config/recipe.json') as recipe:
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
    best_model_name = None
    for model in reports:
        if model['mse'] < lowest_mse:
            lowest_mse = model['mse']
            best_model_name = model['model']  

    return best_model_name, lowest_mse


def sanity_check():
    test_recipe = ["a.json", "b.json", "c.json"]
    a = {"Dog": 1, "Cat": 2, "Giraffe": 3}
    b = {"Fermat": 4, "Euler": 5, "Galois": 6}
    c = {"Berkeley": 1, "Stanford": 2, "Sunshine": 3}

    # to_json_file(test_recipe, "config", "recipe.json")

    to_json_file(a, "config", "a.json")
    to_json_file(b, "config", "b.json")
    to_json_file(c, "config", "c.json")
    abcd = from_recipe()

def normalizeImage(matrix):
    return preprocessing.scale(matrix)

def zca(X):
    print (X.shape)
    mean_X = X.mean(axis=0)
    num_data = X.shape[0]
    for i in range(num_data):
        X[i] -= mean_X
    sigma = X.dot(X.T) / X.shape[1]
    U,S,V = linalg.svd(sigma)
    epsilon = 1e-5

    xPCAWhite = np.diag(1.0 / np.sqrt(S+epsilon)).dot(U.T).dot(X)
    xZCAWhite = U.dot(xPCAWhite)
    return xZCAWhite

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
