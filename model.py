"""The script used to create and train the model."""
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from scipy.misc import imread
from sklearn.model_selection import ShuffleSplit

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

splitter = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
for train_index, test_index in splitter.split(image_urls):
    print(train_index)
    path_train, angle_train = image_urls[train_index], angles[train_index]
    path_test, angle_test = image_urls[test_index], angles[test_index]


def generate_data(paths, angles, batch_size=128):
    """Generator that generates batch data from paths and angles"""
    total_size = len(paths)
    for offset in range(0, batch_size, total_size):
        stop = offset + batch_size
        batch_paths = paths[offset:stop]
        _X_batch = [imread(path).astype(np.float32) for path in batch_paths]
        _y_batch = angles[offset:stop]
        yield _X_batch, _y_batch

# TODO(Olala): define model here
for i in range(3):
    gen = generate_data(path_train, angle_train, batch_size=5)
    X_batch, y_batch = next(gen)
    for X, y in zip(X_batch, y_batch):
        print(X, y)

# TODO(Olala): train model here

# TODO(Olala): save model to model.h5

