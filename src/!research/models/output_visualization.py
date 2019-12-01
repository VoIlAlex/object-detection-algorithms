from src import *
from data.references import DATASETS


import torch
import torchvision
import math
import numpy as np
import matplotlib.pyplot as plt


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


def visualize(output):
    for output_sample in output:
        channels = output_sample.shape[0]

        # number of subplots
        x_axis, y_axis = 1, channels
        x_temp = 1
        for _ in range(channels):
            x_temp += 1
            y_temp = int(channels // x_temp)
            if abs(x_temp - y_temp) < abs(x_axis - y_axis) and (x_axis * y_axis) <= (x_temp * y_temp):
                x_axis = int(x_temp)
                y_axis = int(y_temp)

        for plot_idx in range(channels):
            plt.subplot(x_axis, y_axis, plot_idx + 1)
            sample = output_sample[plot_idx].detach().numpy()
            sample = sample + sample.min()
            sample = sample * (255 / sample.max())
            sample = sample.astype('uint32')
            plt.imshow(sample)

        plt.show()


if __name__ == "__main__":
    input_shape = (3, 100, 100)
    output_shape = (4, 100, 100)
    count = 10

    layer = get_layer(input_shape, output_shape)
    input = get_input(count, input_shape)
    output = get_output_from_layer(layer, input)
    visualize(output)
