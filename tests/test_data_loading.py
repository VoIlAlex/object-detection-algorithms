import os
import sys

# make root dir visible
# for importing
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)


from data.data_loading import *
from src.utils.testutils import *
from data.references import DATASETS

import numpy as np


@allow_MoyaPizama_constrains
class TestDataLoading:

    def test_name_parsing(self):
        result = CocoStyleDataGenerator._CocoStyleDataGenerator__parse_name(
            'COCO_train2014_000000000034.txt'
        )
        dataset_name, partiton_name, unique_number = result

        assert dataset_name == 'COCO'
        assert partiton_name == 'train2014'
        assert unique_number == 34

    def test_names_filtering(self):
        image_names = [
            'COCO_train2014_000000581860.jpg',
            'COCO_train2014_000000581873.jpg',
            'COCO_train2014_000000581880.jpg',
            'COCO_train2014_000000581881.jpg',
            'COCO_train2014_000000581882.jpg',
            'COCO_train2014_000000581884.jpg',
            'COCO_train2014_000000581900.jpg',
            'COCO_train2014_000000581903.jpg',
            'COCO_train2014_000000581904.jpg',
            'COCO_train2014_000000581906.jpg',
            'COCO_train2014_000000581909.jpg',
            'COCO_train2014_000000581921.jpg'
        ]

        label_names = [
            'COCO_train2014_000000581821.txt',
            'COCO_train2014_000000581835.txt',
            'COCO_train2014_000000581839.txt',
            'COCO_train2014_000000581857.txt',
            'COCO_train2014_000000581860.txt',
            'COCO_train2014_000000581873.txt',
            'COCO_train2014_000000581880.txt',
            'COCO_train2014_000000581881.txt',
            'COCO_train2014_000000581882.txt',
            'COCO_train2014_000000581884.txt',
            'COCO_train2014_000000581900.txt',
            'COCO_train2014_000000581903.txt',
            'COCO_train2014_000000581904.txt',
            'COCO_train2014_000000581906.txt'
        ]

        expected_image_names_result = [
            'COCO_train2014_000000581860.jpg',
            'COCO_train2014_000000581873.jpg',
            'COCO_train2014_000000581880.jpg',
            'COCO_train2014_000000581881.jpg',
            'COCO_train2014_000000581882.jpg',
            'COCO_train2014_000000581884.jpg',
            'COCO_train2014_000000581900.jpg',
            'COCO_train2014_000000581903.jpg',
            'COCO_train2014_000000581904.jpg',
            'COCO_train2014_000000581906.jpg'
        ]

        expected_label_names_result = [
            'COCO_train2014_000000581860.txt',
            'COCO_train2014_000000581873.txt',
            'COCO_train2014_000000581880.txt',
            'COCO_train2014_000000581881.txt',
            'COCO_train2014_000000581882.txt',
            'COCO_train2014_000000581884.txt',
            'COCO_train2014_000000581900.txt',
            'COCO_train2014_000000581903.txt',
            'COCO_train2014_000000581904.txt',
            'COCO_train2014_000000581906.txt'
        ]

        image_names_result, label_names_result \
            = CocoStyleDataGenerator.\
            _CocoStyleDataGenerator__filter_names(
                image_names=image_names,
                label_names=label_names
            )

        assert image_names_result == expected_image_names_result
        assert label_names_result == expected_label_names_result

    @MoyaPizama_specific_method
    def test_parse_class_and_bounding_box(self):
        test_generator = CocoStyleDataGenerator(
            images_dir=DATASETS['coco']['test']['images'],
            labels_dir=DATASETS['coco']['test']['labels'],
            names_path=DATASETS['coco']['names']
        )

        # I chose the file
        # randomly from the
        # dataset
        labels = parse_class_and_bounding_box(
            label_path=os.path.join(
                test_generator._labels_dir,
                'COCO_val2014_000000578655.txt'
            ),
            classes_count=80
        )

        # 3 object detected
        # 80 class outputs & 4 coordinates
        # of the bounding box
        assert labels.shape == (3, 84)

        # The labels below are
        # actual labels from the
        # file

        # first label
        assert labels[0, 33] == 1
        assert all(labels[0, 0:33] == 0)
        assert all(labels[0, 34:80] == 0)

        # second label
        assert labels[1, 0] == 1
        assert all(labels[1, 1:80] == 0)

        # third label
        assert labels[2, 0] == 1
        assert all(labels[2, 1:80] == 0)

    @MoyaPizama_specific_method
    def test_default_image_parsing(self):
        test_generator = CocoStyleDataGenerator(
            images_dir=DATASETS['coco']['test']['images'],
            labels_dir=DATASETS['coco']['test']['labels'],
            names_path=DATASETS['coco']['names']
        )

        img = test_generator._CocoStyleDataGenerator__parse_img(
            img_name='COCO_val2014_000000580704.jpg'
        )

        assert isinstance(img, np.ndarray)

    @allow_MoyaPizama_constrains
    class TestCocoLoading:

        @MoyaPizama_specific_method
        def test_coco_dataset_loading(self):
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

        @MoyaPizama_specific_method
        def test_coco_train_batch_getting(self):
            test_generator = CocoStyleDataGenerator(
                images_dir=DATASETS['coco']['test']['images'],
                labels_dir=DATASETS['coco']['test']['labels'],
                names_path=DATASETS['coco']['names'],
                image_shape=(448, 448, 3),
                batch_size=32,
                to_fit=True
            )
            X, y = test_generator[0]

        @MoyaPizama_specific_method
        def test_coco_train_batch_img_shape(self):
            test_generator = CocoStyleDataGenerator(
                images_dir=DATASETS['coco']['test']['images'],
                labels_dir=DATASETS['coco']['test']['labels'],
                names_path=DATASETS['coco']['names'],
                image_shape=(448, 448, 3),
                batch_size=32,
                to_fit=True
            )
            X, y = test_generator[0]
            assert X.shape == (32, 448, 448, 3)

        @MoyaPizama_specific_method
        def test_coco_test_batch_getting(self):
            test_generator = CocoStyleDataGenerator(
                images_dir=DATASETS['coco']['test']['images'],
                labels_dir=DATASETS['coco']['test']['labels'],
                names_path=DATASETS['coco']['names'],
                image_shape=(448, 448, 3),
                batch_size=32,
                to_fit=False
            )
            X = test_generator[0]

        @MoyaPizama_specific_method
        def test_coco_test_batch_shape(self):
            test_generator = CocoStyleDataGenerator(
                images_dir=DATASETS['coco']['test']['images'],
                labels_dir=DATASETS['coco']['test']['labels'],
                names_path=DATASETS['coco']['names'],
                image_shape=(448, 448, 3),
                batch_size=32,
                to_fit=False
            )
            X = test_generator[0]
            assert X.shape == (32, 448, 448, 3)
