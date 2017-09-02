import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Convolution2D, Lambda
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils.visualize_util import plot

from datetime import datetime

# Helper function to load data from a directory
def load_data(data_dir, X, y):

    with open(data_dir + '\\driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        headers = next(reader)

        for line in reader:
            center_image_file_path = data_dir + '\\' + line[headers.index('center')]

            image = cv2.imread(center_image_file_path)
            steer = float(line[headers.index('steering')])

            X.append(image)
            y.append(steer)

            # add flipped image to dataset
            X.append(np.fliplr(image))
            y.append(-steer)

    return

def train(X, y):

    # define nvidia model
    model = Sequential()
    model.add(Lambda(lambda x : (x / 255.0) -0.5, input_shape = (160, 320, 3)))
    model.add(Cropping2D(cropping = ((70, 25), (0, 0)), input_shape = (160, 320, 3)))
    model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation = 'relu'))
    model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation = 'relu'))
    model.add(Convolution2D(48, 5, 5, subsample = (2, 2), activation = 'relu'))
    model.add(Convolution2D(64, 3, 3, activation = 'relu'))
    model.add(Convolution2D(64, 3, 3, activation = 'relu'))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss = 'mse', optimizer = 'adam')

    plot(model, to_file='model.png', show_shapes=True)

    # configure early_stopping_callback to prevent overfitting
    checkpoint_callback = ModelCheckpoint('./model_checkpoints/model_{epoch:02d}.h5',  verbose = 1)
    history = model.fit(X, y, validation_split = 0.2, shuffle = True, nb_epoch = 20, callbacks = [checkpoint_callback])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('model_loss.png')

    return

if __name__ == '__main__':

    X = []
    y = []

    print(datetime.now())

    # load data from various directories in X, y arrays
    load_data('.\\data\\sample_data', X, y)
    load_data('.\\data\\lap', X, y)
    load_data('.\\data\\specific_training_1', X, y)
    load_data('.\\data\\specific_training_2', X, y)
    load_data('.\\data\\specific_training_3', X, y)

    print(datetime.now())

    train(np.array(X), np.array(y))

    print(datetime.now())
