import sys
import os

from src import *
from data.references import DATASETS
from pytorch_modelsize import SizeEstimator

import torch.nn as nn
from torch.functional import F


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
        self.l10_1 = get_L10()
        self.l11_1 = get_L11()

        self.l10_2 = get_L10()
        self.l11_2 = get_L11()

        self.l10_3 = get_L10()
        self.l11_3 = get_L11()

        self.l10_4 = get_L10()
        self.l11_4 = get_L11()

        self.l12 = get_L12()
        self.l13 = get_L13()
        self.l14 = get_L14()

        # Block 5
        self.l15_1 = get_L15()
        self.l16_1 = get_L16()

        self.l15_2 = get_L15()
        self.l16_2 = get_L16()

        self.l17 = get_L17()
        self.l18 = get_L18()

        # Block 6
        self.l19 = get_L19()
        self.l20 = get_L20()

        # Block 7
        self.flat = nn.Flatten()
        self.l21 = get_L21()

        # Block 8
        self.l22 = get_L22()
        self.final_reshape = View(30, 7, 7)


if __name__ == "__main__":
    # Don't show errors and warnings
    # (because it's kinda annoying)
    # sys.stderr = open(os.devnull, 'w')

    # Layer to examine
    net = YOLO()

    # Estimate parameters number
    pytorch_total_params = sum(p.numel()
                               for p in net.parameters() if p.requires_grad)

    # Estimate size
    se = SizeEstimator(net, input_size=(1, 3, 448, 448))
    se.get_output_sizes()

    # Visualize the results

    # Shapes of outputs
    for i, out_size in enumerate(se.out_sizes, 1):
        print("Output of the layer #{}: {}".format(i, out_size))

    print('Size of the layer in megabytes: {}'.format(se.estimate_size()[0]))
    print('Number of parameters: {}'.format(pytorch_total_params))
