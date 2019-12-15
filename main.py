"""
    Works for keras.
"""

from src import *
from data.data_loading import VOCDataGenerator
from data.references import DATASETS
import torch
from src.utils.path_generation import ModelPathGenerator
import src.config as config

if __name__ == "__main__":
    # train_generator = CocoStyleDataGenerator(
    #     images_dir=DATASETS['coco']['train']['images'],
    #     labels_dir=DATASETS['coco']['train']['labels'],
    #     names_path=DATASETS['coco']['names']
    # )
    # test_generator = CocoStyleDataGenerator(
    #     images_dir=DATASETS['coco']['test']['images'],
    #     labels_dir=DATASETS['coco']['test']['labels'],
    #     names_path=DATASETS['coco']['names']
    # )

    train_generator = VOCDataGenerator(
        images_dir=DATASETS['voc']['images_dir'],
        label_txt="data/labels/voc2007.txt",
        image_shape=(448, 448, 3),
        grid_size=7,
        max_bb_count=2,
        num_classes=20,
        to_fit=True
    )

    test_generator = VOCDataGenerator(
        images_dir=DATASETS['voc']['images_dir'],
        label_txt="data/labels/voc2007.txt",
        image_shape=(448, 448, 3),
        grid_size=7,
        max_bb_count=2,
        num_classes=20,
        to_fit=False
    )

    model = YOLO(B=2, S=7, C=20)
    model.fit_generator(
        generator=train_generator,
        validation_data=test_generator,
        channels_first=True
    )

    path_generator = ModelPathGenerator(config.MY_PRETRAINED_MODELS_PATH)
    path = path_generator.generate_path(
        model=model,
        session=1
    )
    torch.save(YOLO.state_dict(), path)
