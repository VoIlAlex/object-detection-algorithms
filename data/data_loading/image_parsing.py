import cv2
import numpy as np


class ImageParser:
    """
        Base class of all the image parsers.

    Raises:
        NotImplementedError: if parse_image(...) isn't implemented.
    """

    def __init__(self):
        pass

    def parse_image(self, image_path: str):
        """Parses label into NN compatible format.

        Arguments:
            image_path {str} -- path to the image to parse
        """
        raise NotImplementedError(
            'parse_image method should be'
            'implemented in child-class of ImageParser'
        )


class DefaultImageParser(ImageParser):
    """
        Default image parser.
    """

    def __init__(self, image_shape: tuple):
        """

        Arguments:
            image_shape {tuple} -- shape of a resultant image
        """
        self._image_shape = image_shape

        # call the parent constructor
        ImageParser.__init__(self)

    def parse_image(self, image_path: str):
        """Parses image into NN compatible format.

        Arguments:
            image_path {str} -- path to the image to parse

        Returns:
            [type] -- [description]
        """
        img = cv2.imread(image_path)
        img = cv2.resize(img, dsize=self._image_shape[:2])
        return img
