from src import *

from keras.preprocessing.image import ImageDataGenerator
from data.references import DATASETS


class X:
    def __init__(self, X, y):
        pass


if __name__ == "__main__":

    generator = ImageDataGenerator()
    data_iterator = generator.flow_from_directory(
        directory=DATASETS['coco']['train']['images']
    )
    print(data_iterator)
