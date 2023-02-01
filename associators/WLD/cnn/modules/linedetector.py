from typing import List, Tuple

import cv2
import numpy as np

Keylines = List[cv2.line_descriptor_KeyLine]
Descriptors = List[np.ndarray]


class LineDetector:
    def set_width_of_band(self, cutout_width: int = 27) -> None:
        # set width of band based on cutout width
        cutout_width = int(cutout_width)
        self.__bd.setWidthOfBand(int(cutout_width / 9))

    def detect(self, image: np.ndarray, min_length: int = 15) -> Keylines:
        # detect keylines using LSDDetector while using same parameters as BinaryDescriptor
        scale = int(self.__bd.getReductionRatio())
        num_octaves = int(self.__bd.getNumOfOctaves())
        keylines = self.__lsd.detect(image, scale, num_octaves)

        # filter out keylines that do not meet the minimum length criteria
        keylines = [kl for kl in keylines if np.float_(kl.lineLength) >= np.float_(min_length)]

        return keylines

    def compute(self, image: np.ndarray, keylines: Keylines) -> Tuple[Keylines, Descriptors]:
        # compute descriptors using given image and keylines
        keylines, descriptors = self.__bd.compute(image, keylines)

        # convert numpy array to list
        descriptors = list(descriptors)

        return keylines, descriptors

    def __init__(self, num_of_octaves: int = 1, reduction_ratio: int = 2, width_of_band: int = 3) -> None:
        # create pointer to LSDDetector
        lsd = cv2.line_descriptor.LSDDetector_createLSDDetector()

        # create pointer to BinaryDescriptor and set parameters
        bd = cv2.line_descriptor.BinaryDescriptor_createBinaryDescriptor()
        bd.setNumOfOctaves(int(num_of_octaves))
        bd.setReductionRatio(int(reduction_ratio))
        bd.setWidthOfBand(int(width_of_band))

        # set instance variables
        self.__lsd = lsd
        self.__bd = bd
