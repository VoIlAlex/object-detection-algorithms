"""
    I don't have enough GPU for the model.
    My GPU is enough only for:
        - loading the model
        - loading the data
        - go through 3 layers
"""

import sys
import os

from src import *
from data.references import DATASETS
from data.data_loading import CocoStyleDataGenerator
from pytorch_modelsize import SizeEstimator

import torch
import torch.nn as nn
from torch.functional import F
import torch.optim as optim

import matplotlib.pyplot as plt


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = (-1,) + shape

    def forward(self, input):
        return input.view(*self.shape)

# Layers of the YOLO networks


def get_L1():
    return nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=(7, 7),
        stride=2,
        padding=3  # ? is padding necessary
    )


def get_L2():
    return nn.MaxPool2d(
        kernel_size=(2, 2),
        stride=2
    )


def get_L3():
    return nn.Conv2d(
        in_channels=64,
        out_channels=192,
        kernel_size=(3, 3),
        padding=1
    )


def get_L4():
    return nn.MaxPool2d(
        kernel_size=(2, 2),
        stride=2
    )


def get_L5():
    return nn.Conv2d(
        in_channels=192,
        out_channels=128,
        kernel_size=(1, 1)
    )


def get_L6():
    return nn.Conv2d(
        in_channels=128,
        out_channels=256,
        kernel_size=(3, 3),
        padding=1
    )


def get_L7():
    return nn.Conv2d(
        in_channels=256,
        out_channels=256,
        kernel_size=(1, 1)
    )


def get_L8():
    return nn.Conv2d(
        in_channels=256,
        out_channels=512,
        kernel_size=(3, 3),
        padding=1
    )


def get_L9():
    return nn.MaxPool2d(
        kernel_size=(2, 2),
        stride=2
    )


def get_L10():
    return nn.Conv2d(
        in_channels=512,
        out_channels=256,
        kernel_size=(1, 1)
    )


def get_L11():
    return nn.Conv2d(
        in_channels=256,
        out_channels=512,
        kernel_size=(3, 3),
        padding=1
    )


def get_L12():
    return nn.Conv2d(
        in_channels=512,
        out_channels=512,
        kernel_size=(1, 1)
    )


def get_L13():
    return nn.Conv2d(
        in_channels=512,
        out_channels=1024,
        kernel_size=(3, 3),
        padding=1
    )


def get_L14():
    return nn.MaxPool2d(
        kernel_size=(2, 2),
        stride=2
    )


def get_L15():
    return nn.Conv2d(
        in_channels=1024,
        out_channels=512,
        kernel_size=(1, 1)
    )


def get_L16():
    return nn.Conv2d(
        in_channels=512,
        out_channels=1024,
        kernel_size=(3, 3),
        padding=1
    )


def get_L17():
    return nn.Conv2d(
        in_channels=1024,
        out_channels=1024,
        kernel_size=(3, 3),
        padding=1
    )


def get_L18():
    return nn.Conv2d(
        in_channels=1024,
        out_channels=1024,
        kernel_size=(3, 3),
        stride=2,
        padding=1
    )


def get_L19():
    return nn.Conv2d(
        in_channels=1024,
        out_channels=1024,
        kernel_size=(3, 3),
        padding=1
    )


def get_L20():
    return nn.Conv2d(
        in_channels=1024,
        out_channels=1024,
        kernel_size=(3, 3),
        padding=1
    )


def get_L21():
    return nn.Linear(
        in_features=50176,
        out_features=4096
    )


def get_L22():
    return nn.Linear(
        in_features=4096,
        out_features=1470
    )


class YOLO(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        # Block 1
        self.l1 = get_L1()
        self.l2 = get_L2()

        # Block 2
        self.l3 = get_L3()
        self.l4 = get_L4()

        # Block 3
        self.l5 = get_L5()
        self.l6 = get_L6()
        self.l7 = get_L7()
        self.l8 = get_L8()
        self.l9 = get_L9()

        # Block 4
        # low_gpu #self.l10_1 = get_L10()
        # low_gpu #self.l11_1 = get_L11()
# low_gpu #
        # low_gpu #self.l10_2 = get_L10()
        # low_gpu #self.l11_2 = get_L11()
# low_gpu #
        # low_gpu #self.l10_3 = get_L10()
        # low_gpu #self.l11_3 = get_L11()
# low_gpu #
        # low_gpu #self.l10_4 = get_L10()
        # low_gpu #self.l11_4 = get_L11()

        self.l12 = get_L12()
        self.l13 = get_L13()
        self.l14 = get_L14()

        # Block 5
        self.l15_1 = get_L15()
        self.l16_1 = get_L16()

        # low_gpu # self.l15_2 = get_L15()
        # low_gpu # self.l16_2 = get_L16()

        self.l17 = get_L17()
        self.l18 = get_L18()

        # Block 6
        self.l19 = get_L19()
        self.l20 = get_L20()

        # Block 7
        self.flat = nn.Flatten()
        # low_gpu # self.l21 = get_L21()

        # Block 8
        # right_yolo # self.l22 = get_L22()
        # right_yolo # self.final_reshape = View(30, 7, 7)

        # low_gpu # self.l22 = nn.Linear(
        # low_gpu #     in_features=4096,
        # low_gpu #     out_features=8400
        # low_gpu # )
        self.view_1 = View(1, 50176)
        self.max_pull_at_end = nn.MaxPool1d(kernel_size=8, stride=8)
        self.flatten_2 = nn.Flatten()
        self.l22 = nn.Linear(
            in_features=6272,
            out_features=8400
        )
        self.final_reshape = View(100, 84)

    def forward(self, x):
        for i, m in enumerate(self.modules()):
            if i != 0:
                x = F.relu(m(x))
        return x


if __name__ == "__main__":
    # Don't show errors and warnings
    # (because it's kinda annoying)
    sys.stderr = open(os.devnull, 'w')

    # Layer to examine

    train_generator = CocoStyleDataGenerator(
        images_dir=DATASETS['coco']['train']['images'],
        labels_dir=DATASETS['coco']['train']['labels'],
        names_path=DATASETS['coco']['names'],
        batch_size=1
    )
    test_generator = CocoStyleDataGenerator(
        images_dir=DATASETS['coco']['test']['images'],
        labels_dir=DATASETS['coco']['test']['labels'],
        names_path=DATASETS['coco']['names'],
        batch_size=1
    )

    net = YOLO().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # save model
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_generator, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = torch.from_numpy(inputs)
            inputs = inputs.permute(0, 3, 1, 2).float().cuda()
            labels = torch.from_numpy(labels).float().cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            del outputs, labels
            torch.cuda.empty_cache()

            optimizer.step()

            # print statistics\
            running_loss += loss.item()
            if i % 10 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
