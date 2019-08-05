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
from keras.models import Sequential
from keras.layers import Dense, Flatten, InputLayer
import numpy as np
import os
import pandas as pd
import random
import json
import keras
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score

from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

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
    this_expected_labels = []

    for i in range(number_of_labels):
        this_data[labels[i]] = os.listdir("./data/" + id_of_labels[labels[i]] + "/")
        this_test_images[labels[i]] = []

        for j in range(len(this_data[labels[i]]) // 4):
            # print(labels[i], len(data[labels[i]]))
            temp_flower = random.randint(0, len(this_data[labels[i]]) - 1)
            this_expected_labels.append(labels[i])

            this_test_images[labels[i]].append(this_data[labels[i]][temp_flower])
            # remove images added to the test images
            del this_data[labels[i]][temp_flower]
    return this_data, this_test_images, this_expected_labels


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
data, test_images, expected_labels = initialize_images_data(nber_labels)

features = extract_features()

# convert features to data frame (from dict)
trainY = list()  # labels
trainX = list() # pd.DataFrame()  # images data

for label, feats in features.items():
    for i in range(len(feats)):
        trainY.append([label]) # temp_label, ignore_index=True)
    # temp_df = pd.DataFrame(feats)
    # temp_label = pd.DataFrame(label)

    # temp_df['label'] = label
    # dataset = dataset.append(temp_df, ignore_index=True)
    # trainX.append(feats)
    trainX.append(feats)
    # trainX = trainX.append(temp_df, ignore_index=True)

# dataset.head()
# trainX.head()
# trainY = pd.DataFrame(trainY)
# trainY = keras.utils.to_categorical(trainY)

# trainY.head()
# print(len(trainX))
# print(type(trainY))


# get test images features
test_features = []
test_images_paths = []
for test_label, test_flowers in test_images.items():
    basepath = "./data/" + id_of_labels[test_label]

    for test_flower in test_flowers:
        test_flower_path = basepath + "/" + test_flower
        test_images_paths.append(test_flower_path)
        test_feats = get_features(test_flower_path, vgg19_model)
        # flatten the features
        test_features.append(test_feats.flatten())
        # test_features.append(test_feats.squeeze())

# y = dataset.label
# x = dataset.drop('label', axis=1)

# classifier model
# model = MLPClassifier(hidden_layer_sizes=(10, 100))
# model = Sequential()
# model.add(Dense(100, activation='softmax', name='fc2'))
# model.add(Dense(len(labels), kernel_initializer="uniform", activation='softmax'))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# pipeline = Pipeline([('low_variance_filter', VarianceThreshold()), ('model', model)])
# pipeline.fit(trainX, keras.utils.to_categorical(trainY))

classifier = Sequential()

# n_timesteps, n_features, n_outputs = (trainX).shape[0], (trainX).shape[1], (trainY).shape[0]
# print(n_timesteps, n_features, n_outputs)
#
# classifier.add(InputLayer(input_shape=(trainX.shape[1],)))

classifier.add(Conv1D(filters=64, kernel_size=2, activation='relu', name='conv1D1'))
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
classifier.add(Dense(max(labels)+1, name='ds3', activation='softmax'))  # softmax  sigmoid
classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) # rmsprop adam


classifier.fit(trainX, trainY)
score = classifier.evaluate(test_features, expected_labels)
print(score)


pipeline = Pipeline([('low_variance_filter', VarianceThreshold()), ('model', classifier)])
# pipeline.set_params(steps_per_epoch=800, epochs=10)
pipeline.fit(trainX, trainY)  #  , steps_per_epoch=80, epochs=10, batch_size=128)

# print(type(list(trainX)))
# classifier.fit({'input': list(trainX)}, {'targets': keras.utils.to_categorical(trainY)}, steps_per_epoch=800, epochs=10)
               # validation_data=(test_features.shape, expected_labels.shape), steps_per_epoch=800, epochs=10)

# another model
# verbose, epochs, batch_size = 0, 10, 32
# print(trainX.shape)
# n_timesteps, n_features, n_outputs = trainX.shape[0], trainX.shape[1], trainY.shape[0]
# model = Sequential()
# model.add(Flatten())
# model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features), name='conv1'))
# model.add(Conv1D(filters=64, kernel_size=3, activation='relu', name='conv2'))
# model.add(Dropout(0.5))
# model.add(MaxPooling1D(pool_size=2))
#
# model.add(Dense(100, activation='relu'))
#
# model.add(Dense(n_outputs, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# # fit network
# pipeline2 = Pipeline([('low_variance_filter', VarianceThreshold()), ('model', model)])
# pipeline2.fit(trainX, trainY)  #  , epochs=epochs, batch_size=batch_size, verbose=verbose)
# # evaluate model
# _, accuracy = pipeline2.predict(test_features, expected_labels, batch_size=batch_size, verbose=0)
# print(accuracy, accuracy)



# print(pipeline.(test_features))
predicted_labels = pipeline.predict(test_features)
print(predicted_labels)

# detection process
# predicted_labels = pipeline.predict(test_features)


# loss = log_loss(expected_labels, predicted_labels)
# print("loss:", loss)
# brier_loss = brier_score_loss(expected_labels, predicted_labels)
# print("brier", brier_loss)
# calculate roc curve
fpr, tpr, thresholds = roc_auc_score (expected_labels, predicted_labels)


the_score = accuracy_score(expected_labels, predicted_labels)
print("Accuracy: ", the_score * 100, "%")

f, ax = plt.subplots(1, len(test_images_paths))
for z in range(len(test_images_paths)):
    print(test_images_paths[z], predicted_labels[z])
#     ax[i].imshow(Image.open(test_images_paths[i]).resize((200, 200), Image.ANTIALIAS))
#     ax[i].text(10, 180, preds[i], color='k', backgroundcolor='red', alpha=0.8)
# plt.show()


# if __name__ == '__main__':
#     pass
