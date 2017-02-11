"""The script used to create and train the model."""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Lambda
from keras.callbacks import ModelCheckpoint
from scipy.misc import imread
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from math import e, sqrt, pi

import random
import numpy as np
import csv
import os


random_state = 42
# Preprocess:
# read driving_log.csv and prepare training dataset
images = []
steerings = []
current_dir = os.getcwd()


def gauss(x, mu=0, sigma=0.075):
    a = 1/(sigma*sqrt(2*pi))
    return a*e**(-0.5*(float(x-mu)/sigma)**2)

max_gauss = gauss(0)


def _should_skip_angle(steering, max_drop_rate=0.5):
    steer_skip_rate = gauss(steering)*max_drop_rate/max_gauss
    return random.random() > steer_skip_rate


driving_log = './data/driving_log.csv'
print('Reading data from %s ...' % driving_log)
with open(driving_log, 'r') as f:
    csv_reader = csv.reader(f, skipinitialspace=True)
    headers = next(csv_reader)
    print('columns:', headers)
    for row in csv_reader:
        (center, left, right, steering, throttle,
            brake, speed) = row
        steering = float(steering)

        if _should_skip_angle(steering):
            continue

        images.append(center)
        steerings.append(steering)

        recovery = 6./25.
        images.append(left)
        steerings.append(steering + recovery)
        images.append(right)
        steerings.append(steering - recovery)


# train test split
print('Train test split ...')
images, steerings = shuffle(images, steerings, random_state=random_state)
paths_train, paths_test, steerings_train, steerings_test = train_test_split(
    images, steerings, test_size=0.2, random_state=random_state)

# prepare testing data
paths_test = np.array(paths_test)
steerings_test = np.array(steerings_test)
assert paths_test.shape[0] == steerings_test.shape[0]
print('validation set size %d' % steerings_test.shape[0])

# prepare training data
paths_train = np.array(paths_train)
steerings_train = np.array(steerings_train)
assert paths_train.shape[0] == steerings_train.shape[0]
print('validation set size %d' % paths_train.shape[0])


# TODO(Olala): avoid skewed data
# TODO(Olala): augment data


def prepare_data(img_path, steering, random_flip=False):
    """ Prepare image data by croping and resiz eimage
    if augment is True, will randomly shift and flip
    image data in order to prevent skewed training set
    """
    abs_path = os.path.join(current_dir, img_path)
    img = imread(abs_path).astype(np.float32)

    # randomly flip image to balance left/right turn
    if random_flip and random.random() > 0.5:
        img = np.fliplr(img)
        steering = -steering

    return img, steering


def batches(paths, steerings, batch_size=128, training=False):
    """Generator that generates data batch by batch
    validating: boolean indicates whether training or validating
    """
    num_paths = paths.shape[0]
    num_steerings = steerings.shape[0]
    assert num_paths == num_steerings

    while True:
        for offset in range(0, num_paths, batch_size):
            X_batch = []
            y_batch = []

            stop = offset + batch_size
            paths_b = paths[offset:stop]
            steerings_b = steerings[offset:stop]

            for i in range(paths_b.shape[0]):
                img, steering = prepare_data(
                    paths_b[i], steerings_b[i], random_flip=training)
                X_batch.append(img)
                y_batch.append(steering)

            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            yield X_batch, y_batch


def _normalize(X):
    a = -0.1
    b = 0.1
    x_min = 0
    x_max = 255
    return a + (X - x_min) * (b - a) / (x_max - x_min)


# Define and compile model
print('Creating model...')
model = Sequential()

# crop image
model.add(Cropping2D(
    cropping=((50, 30), (10, 10)),
    input_shape=(160, 320, 3)))

# normalize rgb data [0~255] to [-1~1]
model.add(Lambda(_normalize))

# 3@40x150
model.add(Convolution2D(24, 5, 5))
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
nb_epochs = 30

print('Start training... batch size %d' % batch_size)
train_generator = batches(
    paths_train, steerings_train, batch_size=batch_size, training=True)
test_generator = batches(paths_test, steerings_test, batch_size=batch_size)

save_checkpoint = ModelCheckpoint('checkpoint.{epoch:02d}.h5', period=5)
model.fit_generator(
    train_generator, train_size, nb_epochs,
    validation_data=test_generator,
    nb_val_samples=test_size,
    callbacks=[save_checkpoint])


# Save trained model
print('Finished! saving model')
model.save('my_model.h5')

