"""
    See model selection in
    the end of this module.
"""

from ..model_template import ObjectDetectionNet
from ..config import FRAMEWORK_TO_USE, IMPORT_ALL
from ..utils.layers import View

import numpy as np

# NN specific modules
if FRAMEWORK_TO_USE == 'keras' or IMPORT_ALL:
    from keras.models import Sequential
    from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Reshape, Activation, ZeroPadding2D
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
        self._model.add(ZeroPadding2D((3, 3)))
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
        self._model.add(ZeroPadding2D((1, 1)))
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

        self._model.add(ZeroPadding2D((1, 1)))
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

        self._model.add(ZeroPadding2D((1, 1)))
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
            self._model.add(ZeroPadding2D((1, 1)))
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
            self._model.add(ZeroPadding2D((1, 1)))
            self._model.add(Conv2D(
                filters=1024,
                kernel_size=(3, 3),
                activation='relu'
            ))
        self._model.add(ZeroPadding2D((1, 1)))
        self._model.add(Conv2D(
            filters=1024,
            kernel_size=(3, 3),
            activation='relu'
        ))

        self._model.add(ZeroPadding2D((1, 1)))
        self._model.add(Conv2D(
            filters=1024,
            kernel_size=(3, 3),
            strides=(2, 2),
            activation='relu'
        ))

        # 6th block of the scheme
        self._model.add(ZeroPadding2D((1, 1)))
        self._model.add(Conv2D(
            filters=1024,
            kernel_size=(3, 3),
            activation='relu'
        ))
        self._model.add(ZeroPadding2D((1, 1)))
        self._model.add(Conv2D(
            filters=1024,
            kernel_size=(3, 3),
            activation='relu'
        ))

        # 7th block of the scheme
        self._model.add(Flatten())
        self._model.add(Dense(4096))

        # 8 block of the scheme
        self._model.add(Dense(30 * 7 * 7))
        self._model.add(Activation('softmax'))
        self._model.add(Reshape(
            target_shape=(30, 7, 7)
        ))

        self._model.compile(
            optimizer=SGD(),
            loss=MSE,
            metrics=[Accuracy(), ])

        # call parent constructor
        ObjectDetectionNet.__init__(self)


class YOLO_pytorch(nn.Module, ObjectDetectionNet):
    def __init__(self, B=2, S=7, C=20, dtype=torch.float):
        # call parents' constructors
        nn.Module.__init__(self)
        ObjectDetectionNet.__init__(self)
        self.dtype = dtype

        # Block 1
        self.l1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=(7, 7),
            stride=2,
            padding=3,
        ).type(self.dtype)
        self.l2 = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=2
        ).type(self.dtype)

        # Block 2
        self.l3 = nn.Conv2d(
            in_channels=64,
            out_channels=192,
            kernel_size=(3, 3),
            padding=1
        ).type(self.dtype)
        self.l4 = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=2
        ).type(self.dtype)

        # Block 3
        self.l5 = nn.Conv2d(
            in_channels=192,
            out_channels=128,
            kernel_size=(1, 1)
        ).type(self.dtype)
        self.l6 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            padding=1
        ).type(self.dtype)
        self.l7 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(1, 1)
        ).type(self.dtype)
        self.l8 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            padding=1
        ).type(self.dtype)

        self.l9 = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=2
        ).type(self.dtype)

        # Block 4
        self.l10_1 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=(1, 1)
        ).type(self.dtype)
        self.l11_1 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            padding=1
        ).type(self.dtype)

        self.l10_2 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=(1, 1)
        ).type(self.dtype)
        self.l11_2 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            padding=1
        ).type(self.dtype)

        self.l10_3 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=(1, 1)
        ).type(self.dtype)
        self.l11_3 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            padding=1
        ).type(self.dtype)

        self.l10_4 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=(1, 1)
        ).type(self.dtype)
        self.l11_4 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            padding=1
        ).type(self.dtype)

        self.l12 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(1, 1)
        ).type(self.dtype)
        self.l13 = nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size=(3, 3),
            padding=1
        ).type(self.dtype)
        self.l14 = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=2
        ).type(self.dtype)

        # Block 5
        self.l15_1 = nn.Conv2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=(1, 1)
        ).type(self.dtype)
        self.l16_1 = nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size=(3, 3),
            padding=1
        ).type(self.dtype)

        self.l15_2 = nn.Conv2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=(1, 1)
        ).type(self.dtype)
        self.l16_2 = nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size=(3, 3),
            padding=1
        ).type(self.dtype)

        self.l17 = nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=(3, 3),
            padding=1
        ).type(self.dtype)
        self.l18 = nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=(3, 3),
            stride=2,
            padding=1
        ).type(self.dtype)

        # Block 6
        self.l19 = nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=(3, 3),
            padding=1
        ).type(self.dtype)
        self.l20 = nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=(3, 3),
            padding=1
        ).type(self.dtype)

        # Block 7
        self.flat = nn.Flatten()
        self.l21 = nn.Linear(
            in_features=50176,
            out_features=4096
        ).type(self.dtype)

        # Block 8
        self.l22 = nn.Linear(
            in_features=4096,
            out_features=(B * 5 + C) * S * S
        ).type(self.dtype)
        self.final_reshape = View(B * 5 + C, S, S)

    def forward(self, x):
        for i, m in enumerate(self.modules()):
            if i != 0:
                x = F.relu(m(x))
        return x

    def predict(self):
        # TODO: YOLO_torch prediction
        pass

    def evaluate(self):
        # TODO: YOLO_torch evaluation
        pass

    def fit(self):
        # TODO: YOLO_torch fitting
        pass

    def predict_generator(self, generator):
        # TODO: YOLO_torch prediction with generator
        pass

    def evaluate_generator(self, generator):
        # TODO: YOLO_torch evaluation with generator
        pass

    def fit_generator(self, generator, validation_data):
        # TODO: YOLO_torch fitting  with generator
        criterion = torch.nn.MSELoss().cuda()
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.001, momentum=0.9)

        for i, (X, y) in enumerate(generator):
            X = torch.from_numpy(X).permute(
                0, 3, 1, 2).type(torch.float16).cuda()
            y = torch.from_numpy(y).type(torch.float16).cuda()

            # preprocess output
            y = y.cpu().permute(0, 2, 1).reshape(8, 85, 10, 10).cuda()

            # Here I'm trying to
            # process
            for m in self.children():
                m.cuda()
                X_old = X
                X = m(X)
                del X_old
                m.cpu()
                torch.cuda.empty_cache()

            y_pred = X

            loss = criterion(y_pred, y)
            loss.backward()

            # RuntimeError: expected device cpu but got device cuda:0
            optimizer.step()

            if i % 99 == 100:
                print('Loss is {}'.format(loss))


# Choosing an implementation
if FRAMEWORK_TO_USE == 'keras':
    YOLO = YOLO_keras
elif FRAMEWORK_TO_USE == 'pytorch':
    YOLO = YOLO_pytorch
elif FRAMEWORK_TO_USE == 'tensorflow':
    YOLO = YOLO_tensorflow
elif FRAMEWORK_TO_USE == 'theano':
    YOLO = YOLO_theano
