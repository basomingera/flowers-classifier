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

from PIL import Image
import matplotlib.pyplot as plt

import umap
from ripser import Rips

DEBUG = False
# JSON_LABEL_PATH = "./flowers.json"
JSON_LABEL_PATH = "./flower_data/cat_to_name.json"

# VGG19
vgg19_weights = './weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
vgg19_model = VGG19(weights=vgg19_weights, include_top=False)

test_images_paths = []

def init():
    pass


def get_labels():
    # read to ensure they are real there
    raw_labels = os.listdir("./flower_data/train/")
    # labels_name = []
    all_labels = []
    for this_label in raw_labels:
        # all_labels.append(labels_of_id[this_label]['id'])
        all_labels.append(labels_of_id[this_label])

    # print("labels", all_labels, raw_labels)
    return all_labels


def initialize_images_data(number_of_labels, directory):
    this_data = {}

    for i in range(number_of_labels):
        this_data[labels[i]] = os.listdir("./flower_data/" + directory + "/" + id_of_labels[labels[i]] + "/")
    return this_data


def get_json_label_id():
    with open(JSON_LABEL_PATH) as json_file:
        json_label = json.load(json_file)
    return json_label


def id_to_label_dict():
    json_label = get_json_label_id()
    flower_dict = {}
    for each_flower in json_label:
        # flower_dict[json_label[each_flower]['id']] = each_flower
        flower_dict[json_label[each_flower]] = each_flower
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


def extract_features(this_data, basedirectory):
    this_features = {}
    this_labels = {}

    # initialize  dict of features
    # TODO use if to check instead of a for loop
    for x in range(nber_labels):
        this_features[id_of_labels[labels[x]]] = []

    for label, flowers in this_data.items():
        basepath = "./flower_data/" + basedirectory + "/" + id_of_labels[label]

        for flower in flowers:
            feats = get_features(basepath + "/" + flower, vgg19_model)

            if basedirectory == "valid":
                test_images_paths.append(basepath + "/" + flower)

            # this_labels.append(int(id_of_labels[label]))
            # flatten the features
            # this_features[id_of_labels[label]].append(feats.flatten())

            # # print(feats.shape)
            feats_reshaped = np.reshape(feats, (feats.shape[2], -1))
            # print(feats_reshaped)
            # # print()
            # # TDA use of UMAP
            # print(feats_reshaped.shape)
            # feats_frame = pd.DataFrame(feats)
            # feats_frame.head()
            feats_umap = umap.UMAP(n_neighbors=5, metric='correlation', min_dist=0.3).fit_transform((feats_reshaped))
            # feats_umap = umap.UMAP().fit_transform(feats_frame)
            this_features[id_of_labels[label]].append(feats_umap.flatten())
            # print(feats_umap.shape)
            # print()
            # this_features[label].append(feats_umap)

            # squeeze the features
            # this_features[label].append(feats.squeeze())
    return this_features

init()
id_of_labels = id_to_label_dict()
labels_of_id = get_json_label_id()
labels = get_labels()
nber_labels = len(labels)

# initialize a dict of labels and the data set
data = initialize_images_data(nber_labels, "train")
test_images = initialize_images_data(nber_labels, "valid")

train_features = extract_features(data, "train")
test_features = extract_features(test_images, "valid")

# convert features to data frame (from dict)
trainY = list()  # labels
trainX = pd.DataFrame()  # images data

for label, feats in train_features.items():
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
testX = []
testY = []

# test_images_paths = []

for test_label, test_flowers in test_features.items():
    # basepath = "./flower_data/" + test_label

    for test_flower in test_flowers:
        # test_flower_path = basepath + "/" + test_flower
        # test_images_paths.append(test_flower_path)

        testX.append(test_flower)
        testY.append(test_label)

    # test_features.append(umap.UMAP(n_neighbors=5, metric='correlation', min_dist=0.3).
    # fit_transform(np.reshape(test_feats, (test_feats.shape[0], -1))))

    # test_features.append(test_feats.squeeze())
    #########################

# y = dataset.label
# x = dataset.drop('label', axis=1)

# classifier model
print("trainX: ", trainX.shape)
model = MLPClassifier(hidden_layer_sizes=(20, 128, 128, 128), activation='identity', solver='adam')  # ‘identity’, ‘logistic’, ‘tanh’, # ‘lbfgs’, ‘adam’
pipeline = Pipeline([('low_variance_filter', VarianceThreshold()), ('model', model)])
pipeline.fit(trainX, trainY)

# print(pipeline.(test_features))
predicted_labels = pipeline.predict(testX)

the_score = accuracy_score(testY, predicted_labels)
print("Accuracy: ", the_score * 100, "%")

f, ax = plt.subplots(1, len(test_images_paths))
for z in range(len(testY)):
    print(test_images_paths[z], predicted_labels[z], testY[z])
    ax[i].imshow(Image.open(test_images_paths[i]).resize((200, 200), Image.ANTIALIAS))
    ax[i].text(10, 180, predicted_labels[i], color='k', backgroundcolor='red', alpha=0.8)
# plt.show()


# if __name__ == '__main__':
#     pass
