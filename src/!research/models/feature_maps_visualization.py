from src import *
from data.references import DATASETS
import cv2


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
import math


def get_layer(input_shape, output_shape, *args, **kwargs):
    """
    Arguments:
        input_shape {} -- [c, w, h]
        output_shape {[type]} -- [c, w, h]

    Returns:
        [type] -- layer
    """
    in_channels = input_shape[0]
    in_width = input_shape[1]
    in_height = input_shape[2]

    out_channels = output_shape[0]
    out_width = output_shape[1]
    out_height = output_shape[2]

    # calculate parameters
    kernel_size = kwargs.get('kernel_size', (5, 5))
    stride_x = math.ceil((in_width - kernel_size[0] + 1) / out_width)
    stride_y = math.ceil((in_height - kernel_size[1] + 1) / out_height)
    padding_x = ((out_width - (in_width - kernel_size[0] + 1) / stride_x)) // 2
    padding_y = (
        (out_height - (in_height - kernel_size[1] + 1)) / stride_y) // 2

    padding_x = int(padding_x)
    padding_y = int(padding_y)

    # construct layer
    layer = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=(stride_x, stride_y),
        padding=(padding_x, padding_y)
    )

    return layer


def get_input(count, shape):
    """

    Arguments:
        shape {[type]} --
        count {[type]} -- [description]
    """
    result_shape = (count,) + shape
    result = torch.randn(result_shape)
    return result


def get_output_from_layer(layer, input):
    output = layer(input)
    return output


def visualize(output, max_channels=None):
    for output_sample in output:
        channels = output_sample.shape[0]
        max_channels = max_channels if max_channels else channels

        # number of subplots
        x_axis, y_axis = 1, min(channels, max_channels)
        x_temp = 1
        for i in range(min(channels, max_channels)):
            x_temp += 1
            y_temp = int(min(channels, max_channels) // x_temp)
            if abs(x_temp - y_temp) < abs(x_axis - y_axis) and (x_axis * y_axis) <= (x_temp * y_temp):
                x_axis = int(x_temp)
                y_axis = int(y_temp)

        for plot_idx in range(min(channels, max_channels)):
            plt.subplot(x_axis, y_axis, plot_idx + 1)
            sample = output_sample[plot_idx].detach().numpy()
            sample = sample + sample.min()
            sample = sample * (255 / sample.max())
            sample = sample.astype('uint32')
            plt.imshow(sample)

        plt.show()


if __name__ == "__main__":
    # input_shape = (3, 100, 100)
    # output_shape = (4, 100, 100)
    # count = 10

    # layer = get_layer(input_shape, output_shape)
    # input = get_input(count, input_shape)
    # output = get_output_from_layer(layer, input)
    # visualize(output)

    # load dataset
    state_dict = torch.load(
        'models/my_pretrained/DenseNet_SGD001_CrossEntropyLoss_100ep_CIFAR100_b16/model.pth')
    model = torchvision.models.DenseNet(num_classes=100).cuda()
    model.load_state_dict(state_dict)

    # img = cv2.imread(
    #     '/home/voilalex/Datasets/VOCdevkit/VOC2007/JPEGImages/009112.jpg')
    # plt.imshow(img)
    # plt.show()
    # img = np.expand_dims(img, 0)
    # img = torch.from_numpy(img).permute(0, 3, 1, 2).float().cuda()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data/datasets', train=True,
                                             download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='./data/datasets', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)

    img, _ = next(iter(trainloader))
    img = img.to('cuda')

    ch = model.children()

    output = next(ch)(img)

    output = output.cpu()

    visualize(output, max_channels=9)
