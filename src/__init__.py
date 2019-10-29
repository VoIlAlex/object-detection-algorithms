"""
This module collects all the
classes and function of the project.
"""

# base class for all
# the detectors
from .model_template import ObjectDetectionNet

# models
from .RCNN import *
from .SSD import *
from .YOLO import *


# other code
from .utils import *
