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
from preprocess import crop_img

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

        # randomly skip 80% dataset with steering angel zero to
        # increase training speed and avoid skew dataset
        drop_rate = 0.85
        if steering == 0.0 and random.random() < drop_rate:
            continue

        img_paths.append(center)
        steerings.append(steering)

        # Add recovery dataset using left, right camerea image
        recovery_steering = 7./25.
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


def prepare_data(img_path, steering, augment=False):
    """ Prepare image data by croping and resiz eimage
    if augment is True, will randomly shift and flip
    image data in order to prevent skewed training set
    """
    # randomly add a horizontal offset on image
    offset = random.randint(-10, 10) if augment else 0
    abs_path = os.path.join(current_dir, 'data', img_path)
    raw_img = imread(abs_path).astype(np.float32)
    img = crop_img(raw_img, offset)
    steering = steering - (0.1/25. * offset)

    # randomly flip image to balance left/right turn
    if augment and random.random() > 0.5:
        img = np.fliplr(img)
        steering = -steering

    return img, steering


def batches(paths, steerings, batch_size=128, training=False):
    """Generator that generates data batch by batch
    validating: boolean indicates whether training or validating
    """
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
                # set augment=True while training
                img, steering = prepare_data(a_path, a_steering, training)
                X_batch.append(img)
                y_batch.append(steering)

            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            yield X_batch, y_batch


# Define and compile model
logging.info('Creating model')
model = Sequential()
# 3@40x150
model.add(Convolution2D(24, 5, 5, input_shape=(40, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))

# 24@18x73
model.add(Convolution2D(36, 5, 5))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))

# 36@7x34
model.add(Convolution2D(48, 5, 5))
model.add(Activation('relu'))

# 48@3x30
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))

# 64@1x28
model.add(Flatten())
model.add(Dropout(0.5))

# 1x1792
model.add(Dense(120))
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
nb_epochs = 30

train_batches = batches(
    paths_train, steerings_train, batch_size=batch_size, training=True)
samples_per_epoch = paths_train.shape[0]
model.fit_generator(train_batches, samples_per_epoch, nb_epochs)

print('Evaluate on testing data')
test_batches = batches(paths_test, steerings_test, batch_size=batch_size)
test_size = paths_test.shape[0]
scores = model.evaluate_generator(test_batches, test_size)
print('loss:', scores[0])

# Save trained model
model.save('my_model.h5')

