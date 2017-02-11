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
images = []
steerings = []
current_dir = os.getcwd()

# TODO(Olala) should not have recovery data in testing set
driving_log = './data/driving_log.csv'
print('Reading data from %s ...' % driving_log)
with open(driving_log, 'r') as f:
    csv_reader = csv.reader(f, skipinitialspace=True)
    headers = next(csv_reader)
    print('columns:', headers)
    for row in csv_reader:
        (center, left, right, steering, throttle,
            brake, speed) = row
        images.append((center, left, right))
        steerings.append(float(steering))

# train test split
print('Train test split ...')
images_train, images_test, steerings_train, steerings_test = train_test_split(
    images, steerings, test_size=0.2, random_state=42)

# prepare testing data
center_img_path_tests = []
for center, left, right in images_test:
    center_img_path_tests.append(center)
paths_test = np.array(center_img_path_tests)
steerings_test = np.array(steerings_test)
print('validation set size %d' % steerings_test.shape[0])

# prepare (augment) training data
paths_train_aug, steerings_train_aug = [], []

for camera_images, steering in zip(images_train, steerings_train):
    # randomly skip 80% dataset with steering angel zero to
    # increase training speed and avoid skew dataset
    drop_rate = 0.9
    if steering == 0.0 and random.random() < drop_rate:
        continue

    center, left, right = camera_images
    paths_train_aug.append(center)
    steerings_train_aug.append(steering)

    # Add recovery dataset using left, right camerea image
    recovery_steering = 6./25.
    paths_train_aug.append(left)
    steerings_train_aug.append(steering + recovery_steering)
    paths_train_aug.append(right)
    steerings_train_aug.append(steering - recovery_steering)

paths_train = np.array(paths_train_aug)
steerings_train = np.array(steerings_train_aug)
print('training set size %d' % steerings_train.shape[0])
print('\n\n')


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
                img, steering = prepare_data(
                    a_path, a_steering, augment=training)
                X_batch.append(img)
                y_batch.append(steering)

            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            yield X_batch, y_batch


# Define and compile model
print('Creating model...')
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
print('Validating traing / testing data size ...')
assert paths_train.shape[0] == steerings_train.shape[0]
assert paths_test.shape[0] == steerings_test.shape[0]
print('Data looks good!')

train_size = paths_train.shape[0]
test_size = paths_test.shape[0]
batch_size = 128
nb_epochs = 1

# TODO(Olala): periodically save model checkpoint for early stopping
print('Start training... batch size %d' % batch_size)
train_batches = batches(
    paths_train, steerings_train, batch_size=batch_size, training=True)
model.fit_generator(train_batches, train_size, nb_epochs)

print('Evaluate on testing data')
test_batches = batches(paths_test, steerings_test, batch_size=batch_size)
model.evaluate_generator(test_batches, test_size)

# Save trained model
model.save('my_model.h5')

