import scipy.misc
import random
import json


xs = []
ys = []

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0
DATADIR = "../rawdata"

# with open('../db/driving_dataset2/data.json') as data_file:
#     data = json.load(data_file)
#
#     for key, value in data.items():
#         xs.append("../db/driving_dataset2/" + key)
#         ys.append(float(value))

with open(DATADIR + "/driving_dataset/data.txt") as f:
    for line in f:
        xs.append(DATADIR + "/driving_dataset_processed/" + line.split()[0])
        ys.append(float(line.split()[1]) * scipy.pi / 180)

#get number of images
num_images = len(xs)

#shuffle list of images
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

train_xs = xs[:int(len(xs) * 0.7)]
train_ys = ys[:int(len(xs) * 0.7)]

val_xs = xs[-int(len(xs) * 0.3):-int(len(xs) * 0.1)]
val_ys = ys[-int(len(xs) * 0.3):-int(len(xs) * 0.1)]

test_xs = xs[-int(len(xs) * 0.1):]
test_ys = ys[-int(len(ys) * 0.1):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)
num_test_images = len(test_xs)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images])[-150:], [66, 200]) / 255.0)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images])[-150:], [66, 200]) / 255.0)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out
