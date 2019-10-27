from src import *

import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Activation, Reshape, Flatten
from keras.optimizers import SGD
from keras.losses import MSE
from keras.metrics import Accuracy

IMAGES_COUNT = 1000
IMAGES_SHAPE = (32, 32, 3)
CLASSES_NUMBER = 10

# (32 + padding * 2 - kernel_size) / stride


def generate_random_images(count: int, shape: tuple, classes_number: int, output_path: str = None):
    """

    Arguments:
        count {int} -- count of images
        shape {tuple} -- shape of images
        classes_number {int} -- number of classes

    Keyword Arguments:
        output_path {str} -- [description] (default: {None})
    """
    images = (np.random.random_sample(
        ((count,) + shape)) * 255).astype('uint8')
    classes = (np.random.random_sample((count,))
               * classes_number).astype('int')
    labels = []
    for cl in classes:
        label = [0] * classes_number
        label[cl] = 1
        labels.append(label)
    labels = np.array(labels).astype('float')
    return images, labels


class ImageClassifier:
    def __init__(self, classes_number: int, batch_size: int = 1):

        self.__model = Sequential((
            Conv2D(
                10,
                kernel_size=(4, 4),
                strides=(1, 1),
                data_format='channels_last',
                input_shape=IMAGES_SHAPE
            ),
            Activation('relu'),
            Conv2D(
                20,
                kernel_size=(3, 3),
                strides=(2, 2),
                data_format='channels_last'
            ),
            Activation('relu'),
            Conv2D(
                40,
                kernel_size=(2, 2),
                strides=(2, 2),
                data_format='channels_last'
            ),
            Activation('relu'),
            Flatten(data_format='channels_last'),
            Dense(100),
            Activation('relu'),
            Dense(classes_number),
            Activation('softmax')
        ))

        self.__model.compile(
            optimizer=SGD(),
            loss=MSE,
            metrics=[Accuracy(), ]
        )

    def fit(self, X, y, **hyperparameters):
        # extracting hyperparamters
        batch_size = hyperparameters.get('batch_size', 1)
        epochs = hyperparameters.get('epochs', 1)
        callbacks = hyperparameters.get('callbacks', None)
        validation_data = hyperparameters.get('validation_data', None)
        self.__model.fit(X, y,
                         batch_size=batch_size,
                         epochs=epochs,
                         callbacks=callbacks,
                         validation_data=validation_data
                         )

        self.__model.summary()

    def evaluate(self, X_test, y_test):
        return self.__model.evaluate(X_test, y_test)

    def predict(self, X):
        predictions = self.__model.predict(X)
        return predictions


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    images, labels = generate_random_images(
        count=IMAGES_COUNT,
        shape=IMAGES_SHAPE,
        classes_number=CLASSES_NUMBER
    )

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2)

    model = ImageClassifier(
        classes_number=CLASSES_NUMBER,
        batch_size=5
    )

    hyperparameters = {
        'batch_size': 5,
        'epochs': 20
    }

    model.fit(X_train, y_train,
              **hyperparameters)
    print(model.evaluate(X_test, y_test))
