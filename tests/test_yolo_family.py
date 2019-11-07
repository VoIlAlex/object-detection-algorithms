2
import os
import sys

# make root dir visible
# for importing
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from data.references import DATASETS
from src.utils.testutils import *

import torch
import keras
import numpy as np

# Here all the networks reside
import src


class TestYolo:
    class TestYolo_pytorch:
        def test_model_creation(self):
            net = src.YOLO_pytorch()

        def test_forward(self):
            net = src.YOLO_pytorch()
            rand_img = torch.randn(
                size=(1, 3, 448, 448),
                dtype=torch.float
            )
            output = net(rand_img)

        def test_output_shape(self):
            net = src.YOLO_pytorch()
            rand_img = torch.randn(
                size=(1, 3, 448, 448),
                dtype=torch.float
            )
            output = net(rand_img)
            assert output.shape == torch.Size([1, 30, 7, 7])

    class TestYolo_keras:
        def test_model_creation(self):
            net = src.YOLO_keras()

        def test_forward(self):
            net = src.YOLO_keras()
            X = np.random.random_sample((1, 448, 448, 3))
            net.predict(X)

        def test_output_shape(self):
            net = src.YOLO_keras()
            X = np.random.random_sample((1, 448, 448, 3))
            y = net.predict(X)
            assert y.shape == (1, 30, 7, 7)
