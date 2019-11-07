from ..model_template import ObjectDetectionNet
from ..config import FRAMEWORK_TO_USE, IMPORT_ALL


# NN specific modules
if FRAMEWORK_TO_USE == 'keras' or IMPORT_ALL:
    from keras.models import Sequential
    from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Reshape, Activation
    from keras.metrics import Accuracy
    from keras.optimizers import SGD
    from keras.losses import MSE

if FRAMEWORK_TO_USE == 'pytorch' or IMPORT_ALL:
    import torch
    import torch.nn as nn
    from torch.functional import F


###################
# Implementations #
###################


class YOLO_keras(ObjectDetectionNet):
    def __init__(self):
        self._model = Sequential()

        # 1st block of the scheme
        self._model.add(Conv2D(
            filters=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            activation='relu'
        ))
        self._model.add(MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2)
        ))

        # 2nd block of the scheme
        self._model.add(Conv2D(
            filters=192,
            kernel_size=(3, 3),
            activation='relu'
        ))
        self._model.add(MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2)
        ))

        # 3rd block of the scheme
        self._model.add(Conv2D(
            filters=128,
            kernel_size=(1, 1),
            activation='relu'
        ))
        self._model.add(Conv2D(
            filters=256,
            kernel_size=(3, 3),
            activation='relu'
        ))
        self._model.add(Conv2D(
            filters=256,
            kernel_size=(1, 1),
            activation='relu'
        ))
        self._model.add(Conv2D(
            filters=512,
            kernel_size=(3, 3),
            activation='relu'
        ))
        self._model.add(MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2)
        ))

        # 4th block of the scheme
        for i in range(4):
            self._model.add(Conv2D(
                filters=256,
                kernel_size=(1, 1),
                activation='relu'
            ))
            self._model.add(Conv2D(
                filters=512,
                kernel_size=(3, 3),
                activation='relu'
            ))
        self._model.add(Conv2D(
            filters=512,
            kernel_size=(1, 1),
            activation='relu'
        ))

        self._model.add(ZeroPadding2D((1, 1)))
        self._model.add(Conv2D(
            filters=1024,
            kernel_size=(3, 3),
            activation='relu'
        ))
        self._model.add(MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2)
        ))

        # 5th block of the scheme
        for i in range(2):
            self._model.add(Conv2D(
                filters=512,
                kernel_size=(1, 1),
                activation='relu'
            ))
            self._model.add(Conv2D(
                filters=1024,
                kernel_size=(3, 3),
                activation='relu'
            ))
        self._model.add(Conv2D(
            filters=1024,
            kernel_size=(3, 3),
            activation='relu'
        ))
        self._model.add(Conv2D(
            filters=1024,
            kernel_size=(3, 3),
            strides=(2, 2),
            activation='relu'
        ))

        # 6th block of the scheme
        self._model.add(Conv2D(
            filters=1024,
            kernel_size=(3, 3),
            activation='relu'
        ))
        self._model.add(Conv2D(
            filters=1024,
            kernel_size=(3, 3),
            activation='relu'
        ))

        # 7th block of the scheme
        self._model.add(Dense(4096))

        # last layer
        self._model.add(Dense(100 * 84))
        self._model.add(Activation('softmax'))
        self._model.add(Reshape(
            target_shape=(100, 84)
        ))

        self._model.compile(
            optimizer=SGD(),
            loss=MSE,
            metrics=[Accuracy(), ])

        # call parent constructor
        ObjectDetectionNet.__init__(self)
