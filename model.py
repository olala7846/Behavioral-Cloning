"""The script used to create and train the model."""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D
from scipy.misc import imread
from sklearn.model_selection import train_test_split
from preprocess import preprocess_img

import random
import numpy as np
import logging
import csv
import os


# Preprocess:
# read driving_log.csv and prepare training dataset
img_paths = []
steerings = []
current_dir = os.getcwd()

with open('./data/driving_log.csv', 'r') as f:
    csv_reader = csv.reader(f, skipinitialspace=True)
    header = next(csv_reader)
    logging.info('header:', header)
    for row in csv_reader:
        (center, left, right, steering, throttle,
            brake, speed) = row
        steering = float(steering)

        # randomly skip 60% dataset with steering angel zero to
        # increase training speed and avoid skew dataset
        drop_rate = 0.6
        if steering == 0.0 and random.random() < drop_rate:
            continue

        img_paths.append(center)
        steerings.append(steering)

        # Add recovery dataset using left, right camerea image
        recovery_steering = 10./25.
        img_paths.append(left)
        steerings.append(steering + recovery_steering)
        img_paths.append(right)
        steerings.append(steering - recovery_steering)

    img_paths = np.array(img_paths)
    steerings = np.array(steerings).astype(np.float32)

# train test split
paths_train, paths_test, steerings_train, steerings_test = train_test_split(
    img_paths, steerings, test_size=0.2)
logging.info('%d training data' % paths_train.shape[0])


def load_img(img_path):
    abs_path = os.path.join(current_dir, 'data', img_path)
    img = imread(abs_path).astype(np.float32)
    return preprocess_img(img)


def batches(paths, steerings, batch_size=128):
    """Generator that generates data batch by batch"""
    while True:
        num_paths = paths.shape[0]
        num_steerings = steerings.shape[0]
        assert num_paths == num_steerings

        for offset in range(0, num_paths, batch_size):
            stop = offset + batch_size
            paths_batch = paths[offset:stop]
            steerings_batch = steerings[offset:stop]

            X_batch = []
            y_batch = []
            for i in range(paths_batch.shape[0]):
                a_path = paths_batch[i]
                a_steering = steerings_batch[i]
                a_img = load_img(a_path)

                # randomly slip half images in order to
                # balance left/right training data
                if random.random() > 0.5:
                    a_img = np.fliplr(a_img)
                    a_steering = -0.1 * a_steering

                X_batch.append(a_img)
                y_batch.append(a_steering)

            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            yield X_batch, y_batch


# Define and compile model
logging.info('Creating model')
model = Sequential()
# 3@40x160
model.add(Convolution2D(24, 5, 5, input_shape=(40, 160, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))

# 24@18x78
model.add(Convolution2D(36, 5, 5))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))

# 36@7x37
model.add(Convolution2D(48, 5, 5))
model.add(Activation('relu'))

# 48@3x33
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))

# 64@1x31
model.add(Flatten())
model.add(Dropout(0.5))

# 1x1984
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(50))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('relu'))

model.add(Dense(1))
model.compile('Adam', 'mse', metrics=['mse'])

# Train model
logging.info('Start training...')
batch_size = 128
nb_epochs = 5

train_batches = batches(paths_train, steerings_train, batch_size=batch_size)
samples_per_epoch = paths_train.shape[0]
test_batches = batches(paths_test, steerings_test, batch_size=batch_size)
test_size = paths_test.shape[0]

model.fit_generator(
    train_batches, samples_per_epoch, nb_epochs,
    validation_data=test_batches,
    nb_val_samples=test_size
)

# Save trained model
model.save('my_model.h5')

