#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on May 15, 2019
@author: Robert BASOMINGERA
@Ajou University
This project is developed and tested with Python3.5 using pycharm on an Ubuntu 16.04 LTS machine
'''

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import decode_predictions
from keras.applications.vgg19 import preprocess_input

# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.inception_v3 import decode_predictions
# from keras.applications.inception_v3 import preprocess_input

# from keras.applications.resnet50 import ResNet50
# from keras.applications.resnet50 import decode_predictions
# from keras.applications.resnet50 import preprocess_input

from keras.preprocessing import image
import numpy as np
import os


def load_images(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    return img_data


def get_features(img_path, model):
    img_data = load_images(img_path)
    img_features = model.predict(img_data)
    return img_features


# Inception
# inception_weights = './weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
# inception_model = InceptionV3(weights=inception_weights, include_top=False)

# resnet50
# resnet50_weights = './weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
# inception_model = ResNet50(weights=resnet50_weights)
# _get_predictions(inception_model)


# VGG19
vgg19_weights = './weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
vgg19_model = VGG19(weights=vgg19_weights, include_top=False)

labels = os.listdir("./data/")
print("labels", labels)
nber_labels = len(labels)

# initialize a dict of labels and the data set
data = {}
for i in range(nber_labels):
    data[labels[i]] = os.listdir("./data/"+labels[i]+"/")


# initialize  dict of features
features = {}
for i in range(nber_labels):
    features[labels[i]] = []


for label, val in data.items():
    # img_path = basepath + "/Banana/" + each
    basepath = "./data/"+label

    for flower in val:
        feats = get_features(basepath + "/" + flower, vgg19_model)
        features[label].append(feats.flatten())

# if __name__ == '__main__':
#     pass
