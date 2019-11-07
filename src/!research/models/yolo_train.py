"""
    Works for keras.
"""

from src import *
from data.data_loading import CocoStyleDataGenerator
from data.references import DATASETS

if __name__ == "__main__":
    train_generator = CocoStyleDataGenerator(
        images_dir=DATASETS['coco']['train']['images'],
        labels_dir=DATASETS['coco']['train']['labels'],
        names_path=DATASETS['coco']['names']
    )
    test_generator = CocoStyleDataGenerator(
        images_dir=DATASETS['coco']['test']['images'],
        labels_dir=DATASETS['coco']['test']['labels'],
        names_path=DATASETS['coco']['names']
    )

    model = YOLO()
    model.fit_generator(
        generator=train_generator,
        validation_data=test_generator
    )
