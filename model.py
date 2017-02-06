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

import numpy as np
import logging
import csv
import os


# read image file path and angles from csv file
image_urls = []
angles = []
with open('./data/driving_log.csv', 'r') as f:
    csv_reader = csv.reader(f)
    header = next(csv_reader)
    logging.info('header:', header)
    for row in csv_reader:
        (img_center, img_left, img_right,
         angle, throttle, break_, speed,) = row
        # TODO(Olala): use data from multiple cameras
        image_urls.append(img_center)
        angles.append(angle)

    image_urls = np.array(image_urls)
    angles = np.array(angles).astype(np.float32)

# train test split
urls_train, urls_test, angles_train, angles_test = train_test_split(
    image_urls, angles, test_size=0.33)
logging.info('%d training data' % urls_train.shape[0])


def normalize(X):
    a = -0.5
    b = 0.5
    x_min = 0
    x_max = 255
    return a + (X - x_min) * (b - a) / (x_max - x_min)


def generate_data(paths, angles, batch_size=128):
    """Generator that generates batch data from paths and angles"""
    while True:
        num_paths = paths.shape[0]
        num_angles = angles.shape[0]
        assert num_paths == num_angles

        for offset in range(0, num_paths, batch_size):
            stop = offset + batch_size
            paths_batch = paths[offset:stop]
            angles_batch = angles[offset:stop]
            X_batch = []
            y_batch = []
            for i in range(paths_batch.shape[0]):
                a_path = paths_batch[i]
                file_path = os.path.join(os.getcwd(), 'data', a_path)
                a_angle = angles_batch[i]
                img_data = imread(file_path).astype(np.float32)

                if i % 2 == 1:
                    img_data = np.fliplr(img_data)
                    a_angle = -0.1 * a_angle

                img_data_normalized = normalize(img_data)
                X_batch.append(img_data_normalized)
                y_batch.append(a_angle)
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)

        yield X_batch, y_batch


# Define and compile model
logging.info('Creating model')
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(160, 320, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Convolution2D(128, 3, 3))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(256))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Dense(1))
model.compile('Adam', 'mse', metrics=['mse'])

# Train model
logging.info('Start training...')
batch_size = 128
nb_epochs = 5

train_gen = generate_data(urls_train, angles_train, batch_size=batch_size)
samples_per_epoch = urls_train.shape[0]
test_gen = generate_data(urls_test, angles_test, batch_size=batch_size)
test_size = urls_test.shape[0]

model.fit_generator(
    train_gen, samples_per_epoch, nb_epochs,
    validation_data=test_gen,
    nb_val_samples=test_size
)

# Save trained model
model.save('my_model.h5')

