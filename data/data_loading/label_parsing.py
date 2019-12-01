"""
    This module contains classes
    and methods for label parsing.
"""
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np


def parse_class_and_bounding_box(label_path: str, classes_count: int = None, one_hot_encoding: bool = True):
    """
        Parses label file with given name.
        Labels are stored in format:
            <class_number> <x> <y> <w> <h>

        Output is in format:
            <class_1_prob> <class_2_prob> ... <class_classes_count_prob> <x> <y> <w> <h>

        Arguments:
            label_path {str} -- path to the label file
            classes_count {int} -- count of the classes in dataset

        Returns:
            np.ndarray -- list of labels from file   
    """
    labels = []
    with open(label_path) as label_file:
        for line in label_file:
            values_from_line = np.array(
                [float(value) for value in line.rstrip().split(' ')], dtype=object
            )
            values_from_line[0] = int(values_from_line[0])

            # first value is categorical (class)
            if one_hot_encoding is True:
                values_from_line.shape = (1, -1)
                onehotencoder = OneHotEncoder(
                    # categories=list(np.arange(self._classes_count))
                    n_values=classes_count,
                    sparse=False
                )
                column_transformer = ColumnTransformer(
                    [('onehotencoder', onehotencoder, [0])],
                    remainder='passthrough'  # non-categorical columns pass through
                )
                values_from_line = column_transformer.fit_transform(
                    values_from_line)
                values_from_line.shape = (-1,)

            # save parsed value
            labels.append(values_from_line)
    return np.array(labels)


class LabelParser:
    """
        Base class of all the label parsers.

    Raises:
        NotImplementedError: if parse_label(...) isn't implemented.
    """

    def __init__(self):
        pass

    def parse_label(self, label_path: str):
        """Parses label into NN compatible format.

        Arguments:
            label_path {str} -- path to the label file

        Raises:
            NotImplementedError: this method should be implemented.
        """
        raise NotImplementedError(
            'parse_label method should be'
            'implemented in child-class of LabelParser'
        )


class DefaultLabelParser(LabelParser):
    """ 
        Default label parser. 
        Beats inequality in prediction count
        through expanding prediction to the 
        max count. 

    """

    def __init__(self):
        # call the parent constructor
        LabelParser.__init__(self)

    def parse_label(self, label_path: str, classes_count: int, max_bb_count: int):
        """Parses label into NN compatible format.

        Arguments:
            label_path {str} -- path to the label file
            classes_count {int} -- count of the classes of dataset
            max_bb_count {int} -- maximum count of bounding boxes in image 

        Returns:
            np.ndarray -- array of labels of objects on the image
        """
        labels = parse_class_and_bounding_box(
            label_path=label_path,
            classes_count=classes_count
        )

        # Permute labels to the right format
        labels = labels.tolist()[0]
        labels = labels[-4:] + labels[:-4]
        labels = labels[:4] + [1.0] + labels[4:]
        labels = np.array([labels])

        result_labels = None

        i = 0
        for label in labels:
            label = np.expand_dims(label, axis=0)
            if result_labels is None:
                result_labels = label
            else:
                result_labels = np.concatenate((result_labels, label))
            i += 1

        # add extra boxes
        # why would I do that???
        # just because I haven't
        # found anything smarter.
        # For now I have no idea how to
        # make YOLO with arbitrary
        # output size.
        if i != max_bb_count:
            extra_labels = np.zeros(
                shape=(max_bb_count - i, classes_count + 5))
            if result_labels is None:
                result_labels = extra_labels
            else:
                result_labels = np.concatenate((result_labels, extra_labels))

        return result_labels.astype('float')


class ClassificationLabelParser(LabelParser):
    """ 
        Parse labels for classification.
        Does not consider bouding box at all. 
    """
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    # ! This class is deprecated.    ! #
    # ! Use it only for fun          ! #
    # ! It's a dataset for detection ! #
    # ! not for classification       ! #
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

    def __init__(self):
        # call the parent constructor
        LabelParser.__init__(self)

    def parse_label(self, label_path: str, classes_count: int, max_bb_count: int):
        """Parses label into NN compatible format.

        Arguments:
            label_path {str} -- path to the label file
            classes_count {int} -- count of the classes of dataset
            max_bb_count {int} -- maximum count of bounding boxes in image 

        Returns:
            np.ndarray -- array of labels of objects on the image
        """
        labels = parse_class_and_bounding_box(
            label_path=label_path,
            classes_count=classes_count,
            one_hot_encoding=False
        )

        return labels[0, 0]
