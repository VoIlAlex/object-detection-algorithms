"""
    See model selection in
    the end of this module.
"""

from ..model_template import ObjectDetectionNet
from ..config import FRAMEWORK_TO_USE, IMPORT_ALL
from ..utils.layers import View


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
    def __init__(self):
        # call parents' constructors
        nn.Module.__init__(self)
        ObjectDetectionNet.__init__(self)

        # Block 1
        self.l1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=(7, 7),
            stride=2,
            padding=3
        )
        self.l2 = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=2
        )

        # Block 2
        self.l3 = nn.Conv2d(
            in_channels=64,
            out_channels=192,
            kernel_size=(3, 3),
            padding=1
        )
        self.l4 = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=2
        )

        # Block 3
        self.l5 = nn.Conv2d(
            in_channels=192,
            out_channels=128,
            kernel_size=(1, 1)
        )
        self.l6 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            padding=1
        )
        self.l7 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(1, 1)
        )
        self.l8 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            padding=1
        )

        self.l9 = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=2
        )

        # Block 4
        self.l10_1 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=(1, 1)
        )
        self.l11_1 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            padding=1
        )

        self.l10_2 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=(1, 1)
        )
        self.l11_2 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            padding=1
        )

        self.l10_3 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=(1, 1)
        )
        self.l11_3 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            padding=1
        )

        self.l10_4 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=(1, 1)
        )
        self.l11_4 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            padding=1
        )

        self.l12 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(1, 1)
        )
        self.l13 = nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size=(3, 3),
            padding=1
        )
        self.l14 = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=2
        )

        # Block 5
        self.l15_1 = nn.Conv2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=(1, 1)
        )
        self.l16_1 = nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size=(3, 3),
            padding=1
        )

        self.l15_2 = nn.Conv2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=(1, 1)
        )
        self.l16_2 = nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size=(3, 3),
            padding=1
        )

        self.l17 = nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=(3, 3),
            padding=1
        )
        self.l18 = nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=(3, 3),
            stride=2,
            padding=1
        )

        # Block 6
        self.l19 = nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=(3, 3),
            padding=1
        )
        self.l20 = nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=(3, 3),
            padding=1
        )

        # Block 7
        self.flat = nn.Flatten()
        self.l21 = nn.Linear(
            in_features=50176,
            out_features=4096
        )

        # Block 8
        self.l22 = nn.Linear(
            in_features=4096,
            out_features=1470
        )
        self.final_reshape = View(30, 7, 7)

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
        pass


# Choosing an implementation
if FRAMEWORK_TO_USE == 'keras':
    YOLO = YOLO_keras
elif FRAMEWORK_TO_USE == 'pytorch':
    YOLO = YOLO_pytorch
elif FRAMEWORK_TO_USE == 'tensorflow':
    YOLO = YOLO_tensorflow
elif FRAMEWORK_TO_USE == 'theano':
    YOLO = YOLO_theano
