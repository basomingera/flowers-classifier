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
from matplotlib import pyplot

# load the images
img = load_img('test2.jpg')
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)

# size_of_image = img.size

# create image data augmentation generator

# Horizontal and Vertical Shift Augmentation
# datagen = ImageDataGenerator(width_shift_range=0.5)
# datagen = ImageDataGenerator(height_shift_range=0.5)

# flip images
# datagen = ImageDataGenerator(horizontal_flip=True)
# datagen = ImageDataGenerator(vertical_flip=True)

# Random Rotation Augmentation between 0 and 90 degrees
# datagen = ImageDataGenerator(rotation_range=90)

# Random Brightness Augmentation
# Values less than 1.0 darken the image, e.g. [0.5, 1.0], whereas values larger than 1.0 brighten
# the image, e.g. [1.0, 1.5], where 1.0 has no effect on brightness.
# datagen = ImageDataGenerator(brightness_range=[0.2,1.0])

# Random Zoom Augmentation
# zoom values less than 1.0 will zoom the image in; values larger than 1.0 will zoom the image out
# e.g. [0.5,0.5] makes the object in the image 50% larger; [1.5, 1.5] makes the object in the image smaller
datagen = ImageDataGenerator(zoom_range=[0.5,1.0])


# prepare iterator
it = datagen.flow(samples, batch_size=1)


# generate samples and plot
for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # generate batch of images
    batch = it.next()
    # convert to unsigned integers for viewing
    image = batch[0].astype('uint8')
    # plot raw pixel data
    pyplot.imshow(image)
# show the figure
pyplot.show()

