YOLO_DEFAULT_IMAGE_SHAPE = (448, 448, 3)  # shape of input image
YOLO_DEFAULT_CELL_SIZE = 7  # size of cell in the grid
YOLO_DEFAULT_BOXES_PER_CELL = 2  # number of boxes per cell


# This constant is used
# throughout label parsing
COCO_MAX_BB_COUNT = 100  # max count of bounding boxes


# Select from [pytorch, tensorflow, keras, theano]
FRAMEWORK_TO_USE = 'pytorch'


# line in main.py
# by default.
DEFAULT_MAIN = [
    'from src import *',
    'from data.references import DATASETS',
    ''
]


# path to scripts
MAIN_PATH = 'main.py'
DEMO_PATH = 'demo.py'


# whether to import
# all the modules or
# import only modules for
# chosen implementation
IMPORT_ALL = True


# paths to models
PRETRAINED_MODELS_PATH = 'models'
MY_PRETRAINED_MODELS_PATH = 'models/my_pretrained'
THIRD_PARTY_PRETRAINED_MODELS_PATH = 'models/pretrained'


# paths to images
DEMO_IMAGES_DIR_PATH = 'data/demo'
