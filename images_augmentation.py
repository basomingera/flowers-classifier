#!/usr/bin/env python3

'''

This file is for image augmentation. A process to artificially
expand the size of images by modifying already stored images.

This program is developed from the tutorial at
https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
'''

# example of horizontal shift image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot, ticker

import os


def get_list_of_images():
    this_data = {}
    labels = os.listdir("./data/")

    for j in range(len(labels)):
        this_data[labels[j]] = os.listdir("./data/" + labels[j] + "/")
    return this_data


class ImagesAugmentation(object):
    def __init__(self, image_path):
        self.image_path = image_path
        # load the images
        self.img = load_img(image_path)

        # convert to numpy array
        self.img_data = img_to_array(self.img)

        # expand dimension to one sample
        self.samples = expand_dims(self.img_data, 0)

    def augmentation_datagen(self):
        # Horizontal and Vertical Shift Augmentation
        aug_datagen_w = ImageDataGenerator(width_shift_range=0.5)
        aug_datagen_h = ImageDataGenerator(height_shift_range=0.5)
        self.process_image(aug_datagen_h, "_augh")
        self.process_image(aug_datagen_w, "_augw")

    # flip images
    def flipImages_datagen(self):
        flip_datagen_h = ImageDataGenerator(horizontal_flip=True)
        flip_datagen_v = ImageDataGenerator(vertical_flip=True)
        self.process_image(flip_datagen_h, "_fliph")
        self.process_image(flip_datagen_v, "_flipv")

    # Random Rotation Augmentation between 0 and 90 degrees
    def random_rot_datagen(self):
        random_datagen = ImageDataGenerator(rotation_range=90)
        self.process_image(random_datagen, "_randrot")

    # Random Brightness Augmentation
    def brightness_datagen(self):
        # Values less than 1.0 darken the image, e.g. [0.5, 1.0], whereas values larger than 1.0 brighten
        # the image, e.g. [1.0, 1.5], where 1.0 has no effect on brightness.
        dark_datagen = ImageDataGenerator(brightness_range=[0.2, 0.9])
        bright_datagen = ImageDataGenerator(brightness_range=[1.0, 2.0])
        self.process_image(bright_datagen, "_bright")
        self.process_image(dark_datagen, "_dark")

    def zoom_datagen(self):
        # Random Zoom Augmentation
        # zoom values less than 1.0 will zoom the image in; values larger than 1.0 will zoom the image out
        # e.g. [0.5,0.5] makes the object in the image 50% larger; [1.5, 1.5] makes the object in the image smaller
        zoomin_datagen = ImageDataGenerator(zoom_range=[0.5, 1.0])
        zoomout_datagen = ImageDataGenerator(zoom_range=[1.1, 2.0])
        self.process_image(zoomin_datagen, "_zoomin")
        self.process_image(zoomout_datagen, "_zoomout")

    def process_image(self, datagen, suffix):
        # prepare iterator
        it = datagen.flow(self.samples, batch_size=1)

        name, extension = os.path.splitext(self.image_path)
        print(name + suffix + extension)

        # generate samples and plot
        for i in range(9):
            # define subplot
            # pyplot.subplot(330 + 1 + i)

            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')

            new_name = name + suffix + str(i) + extension

            pyplot.imshow(image)
            pyplot.axis('off')

            pyplot.gca().xaxis.set_major_locator(ticker.NullLocator())
            pyplot.gca().yaxis.set_major_locator(ticker.NullLocator())

            pyplot.savefig(new_name, bbox_inches='tight', pad_inches=0, transparent="True")
            pyplot.close

        #     pyplot.imshow(image)
        # pyplot.show()


dataset = get_list_of_images()
for label, flowers in dataset.items():
    basepath = "./data/" + label
    for flower in flowers:
        print(basepath + "/" + flower)
        image_augmentation = ImagesAugmentation(basepath + "/" + flower)
        image_augmentation.augmentation_datagen()
        image_augmentation.flipImages_datagen()
        image_augmentation.random_rot_datagen()
        image_augmentation.brightness_datagen()
        image_augmentation.zoom_datagen()
