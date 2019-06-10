#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on May 15, 2019
@author: Robert BASOMINGERA
@Ajou University
This project is developed and tested with Python3.5 using pycharm on an Ubuntu 16.04 LTS machine
'''

from keras.applications.vgg19 import VGG19
# from keras.applications.vgg19 import decode_predictions
from keras.applications.vgg19 import preprocess_input

from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from keras.preprocessing import image
import numpy as np
import os
import pandas as pd
import random
import json
import keras
from sklearn.metrics import accuracy_score

# from PIL import Image
import matplotlib.pyplot as plt

DEBUG = False
JSON_LABEL_PATH = "./flowers.json"

# VGG19
vgg19_weights = './weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
vgg19_model = VGG19(weights=vgg19_weights, include_top=False)


def init():
    pass


def get_labels():
    # read to ensure they are real there
    raw_labels = os.listdir("./data/")
    # labels_name = []
    all_labels = []
    for this_label in raw_labels:
        all_labels.append(labels_of_id[this_label]['id'])

    # print("labels", all_labels, raw_labels)
    return all_labels


def initialize_images_data(number_of_labels):
    this_data = {}
    this_test_images = {}

    for i in range(number_of_labels):
        this_data[labels[i]] = os.listdir("./data/" + id_of_labels[labels[i]] + "/")
        this_test_images[labels[i]] = []

        for j in range(len(this_data[labels[i]]) // 4):
            # print(labels[i], len(data[labels[i]]))
            temp_flower = random.randint(0, len(this_data[labels[i]]) - 1)

            this_test_images[labels[i]].append(this_data[labels[i]][temp_flower])
            # remove images added to the test images
            del this_data[labels[i]][temp_flower]
    return this_data, this_test_images


def get_json_label_id():
    with open(JSON_LABEL_PATH) as json_file:
        json_label = json.load(json_file)
    return json_label


def id_to_label_dict():
    json_label = get_json_label_id()
    flower_dict = {}
    for each_flower in json_label:
        flower_dict[json_label[each_flower]['id']] = each_flower
    return flower_dict


def load_image(img_path):
    if DEBUG:
        print("Opening image:", img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    return img_data


def get_features(img_path, this_model):
    img_data = load_image(img_path)
    img_features = this_model.predict(img_data)
    return img_features


def extract_features():
    this_features = {}

    # initialize  dict of features
    # TODO use if to check instead of a for loop
    for x in range(nber_labels):
        this_features[labels[x]] = []

    for label, flowers in data.items():
        basepath = "./data/" + id_of_labels[label]

        for flower in flowers:
            feats = get_features(basepath + "/" + flower, vgg19_model)
            # flatten the features
            this_features[label].append(feats.flatten())
            # squeeze the features
            # this_features[label].append(feats.squeeze())
    return this_features

init()
id_of_labels = id_to_label_dict()
labels_of_id = get_json_label_id()
labels = get_labels()
nber_labels = len(labels)

# initialize a dict of labels and the data set
data, test_images = initialize_images_data(nber_labels)

features = extract_features()

# convert features to data frame (from dict)
trainY = list()  # labels
trainX = pd.DataFrame()  # images data

for label, feats in features.items():
    for i in range(len(feats)):
        trainY.append([label]) # temp_label, ignore_index=True)
    temp_df = pd.DataFrame(feats)
    # temp_label = pd.DataFrame(label)

    # temp_df['label'] = label
    # dataset = dataset.append(temp_df, ignore_index=True)
    # trainX.append(feats)
    # trainX.append(feats)
    trainX = trainX.append(temp_df, ignore_index=True)

trainX.head()
# trainY = pd.DataFrame(trainY)
# trainY = keras.utils.to_categorical(trainY)

# trainY.head()
# print(len(trainX))
# print(type(trainY))


# get test images features
test_features = []
test_images_paths = []

expected_labels = []

for test_label, test_flowers in test_images.items():
    basepath = "./data/" + id_of_labels[test_label]

    for test_flower in test_flowers:
        test_flower_path = basepath + "/" + test_flower
        test_images_paths.append(test_flower_path)

        expected_labels.append(test_label)

        test_feats = get_features(test_flower_path, vgg19_model)
        # flatten the features
        test_features.append(test_feats.flatten())
        # test_features.append(test_feats.squeeze())

# y = dataset.label
# x = dataset.drop('label', axis=1)

# classifier model
model = MLPClassifier(hidden_layer_sizes=(50, 128, 128, 128), activation='tanh', solver='lbfgs')  # ‘identity’, ‘logistic’, ‘tanh’, # ‘lbfgs’, ‘adam’
pipeline = Pipeline([('low_variance_filter', VarianceThreshold()), ('model', model)])
pipeline.fit(trainX, trainY)

# print(pipeline.(test_features))
predicted_labels = pipeline.predict(test_features)

the_score = accuracy_score(pd.DataFrame(expected_labels), predicted_labels)
print("Accuracy: ", the_score * 100, "%")

f, ax = plt.subplots(1, len(test_images_paths))
for z in range(len(test_images_paths)):
    print(test_images_paths[z], predicted_labels[z], expected_labels[z])
#     ax[i].imshow(Image.open(test_images_paths[i]).resize((200, 200), Image.ANTIALIAS))
#     ax[i].text(10, 180, preds[i], color='k', backgroundcolor='red', alpha=0.8)
# plt.show()


# if __name__ == '__main__':
#     pass
