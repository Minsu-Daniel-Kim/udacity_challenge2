import os
import json


SAVE_DIR = "save"

# Serializes DATA under NAME
# Serialize to save/DIRECTORY/name
# Remember to add extension manually if there needs to be any.
def to_json_file(data, directory, name):
    path_name = SAVE_DIR + '/' + directory + '/' + name
    with open(path_name, 'w') as f:
        json.dump(data, f)


# Opens up a file located in save/DIRECTORY/FILE_NAME
def from_json_file(directory, file_name):
    path_name = SAVE_DIR + '/' + directory + '/' + file_name
    with open(path_name) as data_file:
        data = json.load(data_file)
    print(data)
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
    all_models = from_json_file("report", "report.json")
    for model in all_models:
        if model["model"] == desired_model:
            model["accuracy"] += [entry]
            break
    to_json_file(all_models, "report", "report.json")


# Reads report.json, and finds the best model
# Returns a string (name of best model)
def find_best_model():
    reports = from_json_file("report", "report.json")
    # best_models = [[model1, min_mse], [model2, min_mse]]
    best_models = []
    for model in reports:
        best_models += [[model["model"], model["accuracy"].pop()["acc"]]]
    best_model_name = best_models[0][0]
    best_model_score = best_models[0][1]
    for candidate in best_models:
        if candidate[1] < best_model_score:
            best_model_name = candidate[0]
            best_model_score = candidate[1]
    return best_model_name


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
