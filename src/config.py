IMAGE_SHAPE = (448, 448, 3)  # shape of input image
CELL_SIZE = 7  # size of cell
BOXES_PER_CELL = 2  # number of boxes per cell

# This constant is used
# throughout label parsing
MAX_BB_COUNT = 100  # max count of bounding boxes


FRAMEWORK_TO_USE = 'pytorch'


# line in main.py
# by default.
DEFAULT_MAIN = [
    'from src import *',
    'from data.references import DATASETS',
    ''
]

# path to main script
MAIN_PATH = 'main.py'

# whether to import
# all the modules or
# import only modules for
# chosen implementation
IMPORT_ALL = True
