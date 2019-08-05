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

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, Flatten, Input
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras import optimizers

import numpy as np
import os
import pandas as pd
import random
import json
import keras
from sklearn.metrics import accuracy_score

# from PIL import Image
import matplotlib.pyplot as plt

# import umap

DEBUG = False
JSON_LABEL_PATH = "./flower_data/cat_to_name.json"

# VGG19
vgg19_weights = './weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
vgg19_model = VGG19(weights=vgg19_weights, include_top=False)


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

    # initialize  dict of features
    # TODO use if to check instead of a for loop
    for x in range(nber_labels):
        this_features = []
        this_labels = []

    for label, flowers in this_data.items():
        basepath = "./flower_data/" + basedirectory + "/" + id_of_labels[label]

        for flower in flowers:
            feats = get_features(basepath + "/" + flower, vgg19_model)
            this_features.append(feats)
            this_labels.append(int(id_of_labels[label]))
            # flatten the features
            # this_features[label].append(feats.flatten())

            # # print(feats.shape)
            # feats_reshaped = np.reshape(feats, (feats.shape[0], -1))
            # print(feats_reshaped)
            # # print()
            # # TDA use of UMAP
            # feats_umap = umap.UMAP(n_neighbors=5, metric='correlation', min_dist=0.3).fit_transform(feats_reshaped)
            # print(feats_umap)
            # print()
            # this_features[label].append(feats_umap)

            # squeeze the features
            # this_features[label].append(feats.squeeze())
    return np.array(this_features), np.array(this_labels)

init()
id_of_labels = id_to_label_dict()
labels_of_id = get_json_label_id()
labels = get_labels()
nber_labels = len(labels)

# initialize a dict of labels and the data set
data = initialize_images_data(nber_labels, "train")
test_images = initialize_images_data(nber_labels, "valid")

trainX, trainY = extract_features(data, "train")
# print(trainY)
testX, testY = extract_features(test_images, "valid")


# MLP classifier model
# model = MLPClassifier(hidden_layer_sizes=(100, 128, 128, 128), activation='tanh', solver='adam')  # ‘identity’, ‘logistic’, ‘tanh’, # ‘lbfgs’, ‘adam’
# pipeline = Pipeline([('low_variance_filter', VarianceThreshold()), ('model', model)])
# pipeline.fit(trainX, trainY)
# # print(pipeline.(test_features))
# predicted_labels = pipeline.predict(test_features)
# the_score = accuracy_score(pd.DataFrame(expected_labels), predicted_labels)
# print("Accuracy: ", the_score * 100, "%")

# Neural network classifier
# build a classifier model to put on top of the VGG convolutional model

batch_size = 128
# classifier = Sequential()
# classifier.add(Flatten(input_shape=trainX.shape[1:]))
# classifier.add(Dense(256, activation='relu'))
# classifier.add(Dropout(0.5))
# classifier.add(Dense(len(labels), activation='sigmoid'))


# vgg19_model.add(model)


# vgg19_model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# classifier.compile(loss='binary_crossentropy',
#               optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
#               metrics=['accuracy'])

# print(classifier.summary())

####################################
classifier = Sequential()

n_timesteps, n_features, n_outputs = (trainX).shape[0], (trainX).shape[1], (trainY).shape[0]
print(n_timesteps, n_features, n_outputs)
print(trainX.shape)
print(trainX[0].shape)
#
# classifier.add(InputLayer(input_shape=(trainX.shape[0],)))
# classifier.add(Flatten(input_shape=trainX.shape[1:],))
# classifier.add(Flatten())

classifier.add(Conv1D(64,32, activation='relu', name='conv1D1', input_shape=(1, 7, 7, 512)))
# classifier.add(Conv1D(filters=64, kernel_size=3, activation='relu', batch_input_shape=(0, n_timesteps, n_features), name='conv1D1'))
classifier.add(Conv1D(filters=64, kernel_size=3, activation='relu', name='conv2'))
classifier.add(Dropout(0.5))
classifier.add(MaxPooling1D(pool_size=2))

classifier.add(Dense(128, name='ds0', activation='relu'))  # , input_dim=11))
classifier.add(Dropout(0.3))
classifier.add(Dense(100, activation='relu', name='ds1'))  # , input_shape=(,)))
classifier.add(Dropout(0.3))
classifier.add(Dense(100, activation='softmax', name='ds2'))

classifier.add(Flatten())

classifier.add(Dense(len(labels), name='ds3', activation='softmax'))  # softmax  sigmoid
classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) # rmsprop adam
########################

classifier.fit(trainX, trainY, epochs=50, batch_size=128, validation_data=(testX, testY))

# Final evaluation of the model
scores = classifier.evaluate(testX, testY, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predicted_labels = classifier.predict(testX)
print(predicted_labels)


# f, ax = plt.subplots(1, len(test_images_paths))
for z in range(len(predicted_labels)):
    print(test_images[z], predicted_labels[z], testY)
#     ax[i].imshow(Image.open(test_images_paths[i]).resize((200, 200), Image.ANTIALIAS))
#     ax[i].text(10, 180, preds[i], color='k', backgroundcolor='red', alpha=0.8)
# plt.show()


# if __name__ == '__main__':
#     pass
