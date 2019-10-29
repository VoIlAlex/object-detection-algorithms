import os
import sys
from keras.utils import Sequence
import numpy as np
import cv2
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# make root dir visible
# for importing
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# different types of
# labels parsing
from .label_parsing import LabelParser, DefaultLabelParser
from .image_parsing import ImageParser, DefaultImageParser
import src.config as cfg


class CocoStyleDataGenerator(Sequence):
    # TODO: test CocoStyleDataGenerator in action
    """
    Generates data for Keras models.
    The folders structure should be:

    a) for image folder
    image_folder
    |--- <dataset>_<partition>_<unique_number_1>.jpg
    |--- <dataset>_<partition>_<unique_number_2>.jpg
    ...
    |--- <dataset>_<partition>_<unique_number_n>.jpg

    b) for labels folder
    labels_folder
    |--- <dataset>_<partition>_<unique_number_1>.txt
    |--- <dataset>_<partition>_<unique_number_2>.txt
    ...
    |--- <dataset>_<partition>_<unique_number_n>.txt


    where
        <dataset> - name of the dataset (e.g. COCO)
        <partition> - part of the dataset (e.g. train)
        <unique_number_i> - unique number of the element (pretty obvious, isn't it?)
    """

    def __init__(self,
                 images_dir: str,
                 labels_dir: str,
                 names_path: str,
                 image_shape: tuple = (448, 448, 3),
                 batch_size: int = 32,
                 to_fit: bool = True,
                 shuffle: bool = True,
                 label_parser: LabelParser = None,
                 image_parser: ImageParser = None):
        """

        Arguments:
            images_dir {str} -- directory with images of the dataset
            labels_dir {str} -- directory with labels of the dataset
            names_path {str} -- directory with names of the dataset

        Keyword Arguments:
            image_shape {tuple} -- shape to which image will be transformed (default: {(448, 448, 3)})
            batch_size {int} -- size of one data batch (default: {32})
            to_fit {bool} -- whether to return Y values too (default: {True})
            shuffle {bool} -- whether to shuffle the dataset after each epoch (default: {True})
            label_parser {LabelParser} -- parser of labels for the generator
        """
        self._images_dir = images_dir
        self._labels_dir = labels_dir
        self._image_shape = image_shape
        self._batch_size = batch_size
        self._to_fit = to_fit
        self._shuffle = shuffle
        self._names_path = names_path
        self._classes_count = len(list(open(names_path)))

        # max number of bounding
        # boxes in the image
        self._max_bb_count = cfg.MAX_BB_COUNT

        # the parsers might be
        # customized.
        self._label_parser = label_parser if label_parser else DefaultLabelParser()
        self._image_parser = image_parser if image_parser else DefaultImageParser(
            image_shape)

        image_names = os.listdir(images_dir)
        label_names = os.listdir(labels_dir)

        # ! This area is deprecated

        # if len(images_names) != len(labels_names):
        #     raise ValueError(
        #         'There is no correlation '
        #         'between images count and labels count\n'
        #         'Images count: {}\nLabels count: {}'.format(
        #             len(images_names),
        #             len(labels_names)
        #         )
        #     )

        # leave only labels and
        # images that have common name
        self._image_names, self._label_names = self.__filter_names(
            image_names,
            label_names
        )

        # sort lists
        # ? Does it really necessary
        self._image_names.sort()
        self._label_names.sort()

        self._indexes = []  # it will be filled by the next line
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch

        Returns:
            int -- number of batches per epoch
        """
        return len(self._image_names) // self._batch_size

    def __getitem__(self, index):
        """Generate one batch of data

        Arguments:
            index {int} -- index of the batch
        """

        # Generate indexes of the batch
        indexes = self._indexes[
            index * self._batch_size:(index + 1) * self._batch_size
        ]

        X = None
        for idx in indexes:
            img = self.__parse_img(self._image_names[idx])
            if X is None:
                # reshaping because
                # it's only one image
                # of the batch
                X = np.expand_dims(img, axis=0)

            else:
                # add image to
                # the batch
                X = np.concatenate((X, np.expand_dims(img, axis=0)))

        if self._to_fit:
            y = None
            for idx in indexes:
                label = self.__parse_label(self._label_names[idx])
                if y is None:
                    y = np.expand_dims(label, axis=0)
                else:
                    y = np.concatenate((y, np.expand_dims(label, axis=0)))

            return X, y
        else:
            return X

    def on_epoch_end(self):
        """
            This method will be called
            after each epoch.
        """
        self._indexes = np.arange(len(self._image_names))
        if self._shuffle == True:
            np.random.shuffle(self._indexes)

    def __parse_img(self, img_name: str) -> np.ndarray:
        """Parses image with given name

        Arguments:
            img_name {str} -- name of the image in directory specified in initialization

        Returns:
            np.ndarray -- array of pixels with channels last
        """
        image_path = os.path.join(self._images_dir, img_name)
        img = self._image_parser.parse_image(image_path)
        return img

    def __parse_label(self, label_name: str) -> np.ndarray:
        """
            Parses label file with given name.
            Labels are stored in format:
                <class_number> <x> <y> <w> <h>

        Arguments:
            label_name {str} -- name of the file with labels

        Returns:
            np.ndarray -- list of labels from file
        """
        label_path = os.path.join(self._labels_dir, label_name)
        label = self._label_parser.parse_label(
            label_path=label_path,
            classes_count=self._classes_count,
            max_bb_count=self._max_bb_count)
        return label

    @staticmethod
    def __parse_name(name):
        """
        Split name into parts. 

        Format:
        <dataset>_<partition>_<unique_number_n>.txt

        Arguments:
            name {str} -- name of an element (image or label)

        Returns:
            {tuple(str, str, str)} - dataset name, partition name and unique number
        """
        dataset_name, partition_name, unique_number_raw = name.split('_')
        unique_number = os.path.splitext(unique_number_raw)[0]
        return dataset_name, partition_name, int(unique_number)

    @staticmethod
    def __filter_names(image_names: list, label_names: list):
        """Filter names that are not presented in both lists of names

        Arguments:
            image_names {list} -- list of names of images
            label_names {list} -- list of names of labels 

        Returns:
            tuple -- two filtered lists of images and labels
        """
        # find common names
        image_names_without_ext = set(
            [os.path.splitext(image_name)[0] for image_name in image_names])
        label_names_without_ext = set(
            [os.path.splitext(label_name)[0] for label_name in label_names])
        common_names = image_names_without_ext & label_names_without_ext

        # filter non-common elements
        image_names_result = list(
            filter(lambda name: os.path.splitext(name)[0] in common_names, image_names))
        label_names_result = list(
            filter(lambda name: os.path.splitext(name)[0] in common_names, label_names))

        return image_names_result, label_names_result
