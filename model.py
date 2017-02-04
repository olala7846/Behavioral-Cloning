"""The script used to create and train the model."""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from scipy.misc import imread
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import numpy as np
import csv
import logging


# read image file path and angles from csv file
image_urls = []
angles = []
with open('./driving_log.csv', 'r') as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        (img_center, img_left, img_right,
         angle, throttle, break_, speed,) = row
        # TODO(Olala): use data from multiple cameras
        image_urls.append(img_center)
        angles.append(angle)

    image_urls = np.array(image_urls)
    angles = np.array(angles)

# train test split
urls_train, urls_test, angles_train, angles_test = train_test_split(
    image_urls, angles, test_size=0.33)


def normalize(X):
    a = -0.5
    b = 0.5
    x_min = 0
    x_max = 255
    return a + (X - x_min) * (b - a) / (x_max - x_min)


def generate_data(paths, angles, batch_size=128):
    """Generator that generates batch data from paths and angles"""
    while True:
        total_size = len(paths)
        for offset in range(0, batch_size, total_size):
            stop = offset + batch_size
            batch_paths = paths[offset:stop]
            _X_batch = [imread(path).astype(np.float32) for path in batch_paths]
            _X_batch = np.array(_X_batch)
            _X_batch_normalized = normalize(_X_batch)
            _y_batch = angles[offset:stop]
            yield _X_batch_normalized, _y_batch


# TODO(Olala): define model here
model = Sequential()
model.add(Convolution2D(
    32, 3, 3, border_mode='valid',
    subsample=(2, 2), input_shape=(160, 320, 3)))
model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(2, 2)))
model.add(Convolution2D(128, 3, 3, border_mode='valid', subsample=(1, 2)))
model.add(Flatten())
model.add(Dense(300))
model.add(Dense(150))
model.add(Dense(32))
model.add(Dense(1))
model.compile('Adam', 'mse', metrics=['mse'])

# Train model
batch_size = 128
nb_epochs = 100
gen = generate_data(urls_train, angles_train, batch_size=batch_size)
samples_per_epoch = int(len(urls_train)/batch_size)
model.fit_generator(gen, samples_per_epoch, nb_epochs)

# Save trained model
model.save('my_model.h5')

