import sys
import os

from src import *
from data.references import DATASETS
from pytorch_modelsize import SizeEstimator

import torch.nn as nn
from torch.functional import F

# Layers of the YOLO networks


def get_L1():
    return nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=(7, 7),
        stride=2
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
        kernel_size=(3, 3)
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
        kernel_size=(3, 3)
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
        kernel_size=(3, 3)
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
        kernel_size=(3, 3)
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
        kernel_size=(3, 3)
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
        kernel_size=(3, 3)
    )


def get_L17():
    return nn.Conv2d(
        in_channels=1024,
        out_channels=1024,
        kernel_size=(3, 3)
    )


def get_L18():
    return nn.Conv2d(
        in_channels=1024,
        out_channels=1024,
        kernel_size=(3, 3),
        stride=2
    )


def get_L19():
    return nn.Conv2d(
        in_channels=1024,
        out_channels=1024,
        kernel_size=(3, 3)
    )


def get_L20():
    return nn.Conv2d(
        in_channels=1024,
        out_channels=1024,
        kernel_size=(3, 3)
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


if __name__ == "__main__":
    # Don't show errors and warnings
    # (because it's kinda annoying)
    sys.stderr = open(os.devnull, 'w')

    # Layer to examine
    net = get_L1()

    # Estimate parameters number
    pytorch_total_params = sum(p.numel()
                               for p in net.parameters() if p.requires_grad)

    # Estimate size
    se = SizeEstimator(net, input_size=(1, 3, 448, 448))
    # Visualize the results
    print('Size of the layer in megabytes: {}'.format(se.estimate_size()[0]))
    print('Number of parameters: {}'.format(pytorch_total_params))
