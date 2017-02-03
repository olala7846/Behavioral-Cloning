# TODO(Olala): Train test split (80% 20%)
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from scipy.misc import imread

import numpy as np
import csv
import logging


def train_data_generator(
        batch_size=128, epochs=15, file_path='./driving_log.csv'):
    """Generates training data from csv file and IMAGE folder"""
    for epoch in range(epochs):
        logging.info('start epoch %d', epoch+1)
        with open('./driving_log.csv', 'r') as f:
            csv_reader = csv.reader(f)
            while True:
                X_batch_ = []
                y_batch_ = []
                for i in range(batch_size):
                    (
                        img_center, img_left, img_right,
                        angle, throttle, break_, speed,
                    ) = next(csv_reader)

                    img_data_ = imread(img_center).astype(np.float32)

                    X_batch_.append(img_data_)
                    y_batch_.append(angle)

                yield X_batch_, y_batch_


# TODO(Olala): create and train model here
for i in range(3):
    gen = train_data_generator(batch_size=5)
    X_batch, y_batch = next(gen)
    for X, y in zip(X_batch, y_batch):
        print(X, y)

# TODO(Olala): save model to model.h5

