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
import csv


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
    angles = np.array(angles).astype(np.float32)

# train test split
urls_train, urls_test, angles_train, angles_test = train_test_split(
    image_urls, angles, test_size=0.33)


def normalize(X):
    a = -0.5
    b = 0.5
    x_min = 0
    x_max = 255
    return a + (X - x_min) * (b - a) / (x_max - x_min)


def _gen_single_data(paths, angles):
    """Generate single image angle pair"""
    num_paths = paths.shape[0]
    num_angles = angles.shape[0]
    assert num_paths == num_angles

    while True:
        for i in range(num_paths):
            img_data = imread(paths[i]).astype(np.float32)
            angle = angles[i]
            yield img_data, angle


def generate_data(paths, angles, batch_size=128):
    """Generator that generates batch data from paths and angles"""
    single_data_generator = _gen_single_data(paths, angles)

    while True:
        _X_batch = []
        _y_batch = []
        for i in range(batch_size):
            img_data, angle = next(single_data_generator)
            # flip half of the images to avoid inbalance training data
            if i % 2 == 1:
                img_data = np.fliplr(img_data)
                angle = -1.0 * angle

            # normalize image data into [-0.5~0.5]
            img_normalized = normalize(img_data)
            _X_batch.append(img_normalized)
            _y_batch.append(angle)

        # turns python list into numpy array
        _X_batch = np.array(_X_batch)
        _y_batch = np.array(_y_batch)
        yield _X_batch, _y_batch


# Define and compile model
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
batch_size = 128
nb_epochs = 50
gen = generate_data(urls_train, angles_train, batch_size=batch_size)
samples_per_epoch = int(len(urls_train)/batch_size)
model.fit_generator(gen, samples_per_epoch, nb_epochs)

# Save trained model
model.save('my_model.h5')

