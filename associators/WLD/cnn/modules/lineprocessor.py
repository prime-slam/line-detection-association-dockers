from typing import List, Tuple

import cv2
import numpy as np

from .geometry import Segment2D

Keylines = List[cv2.line_descriptor_KeyLine]
Cutouts = List[np.ndarray]


# https://stackoverflow.com/a/952952
def flatten(nested: List[List[any]]) -> List[any]:
    return [item for sublist in nested for item in sublist]


class LineProcessor:
    def process(self, image: np.ndarray, keylines: Keylines) -> Tuple[Cutouts, List[Cutouts]]:
        # get cutout parameters from current instance
        cutout_width = self.__cutout_width
        cutout_height = self.__cutout_height

        # initialize list of cutouts
        cutouts = []
        wavelet_cutouts = []

        # loop through all keylines
        for keyline in keylines:
            # extract default cutout
            cutout = LineProcessor.__extract_cutout(image, keyline, cutout_width, cutout_height)

            # prepare list of wavelet cutouts
            prepared_cutouts = self.__prepare_wavelet_cutouts(image, keyline, cutout_width, cutout_height)

            # filter each image with each kernel
            gabor_wavelets = self.__filter_each_image_with_each_kernel(prepared_cutouts)

            # flatten nested list of gabor wavelets
            results = flatten(gabor_wavelets)

            # append default cutout and wavelet cutouts to arrays
            cutouts.append(cutout)
            wavelet_cutouts.append(results)

        return cutouts, wavelet_cutouts

    def __init__(self, cutout_width: int = 27, cutout_height: int = 100, num_of_levels: int = 4) -> None:
        # set instance variables
        self.__cutout_width = int(cutout_width)
        self.__cutout_height = int(cutout_height)
        self.__num_of_levels = int(num_of_levels)

        # compute gabor kernels
        self.__gabor_kernels = self.__compute_gabor_kernels()

    def __filter_each_image_with_each_kernel(self, images: List[np.ndarray]) -> List[List[np.ndarray]]:
        # filter each image with each kernel
        results = []
        for image in images:
            filtered_images = []
            for kernel in self.__gabor_kernels:
                filtered_image = cv2.filter2D(np.int16(image), cv2.CV_16S, kernel)
                scaled_filtered_image = np.uint8(cv2.convertScaleAbs(filtered_image, None, 1, 128))
                filtered_images.append(scaled_filtered_image)

            results.append(filtered_images)

        return results

    def __prepare_wavelet_cutouts(self, image: np.ndarray, keyline: cv2.line_descriptor_KeyLine, cutout_width: int,
                                  cutout_height: int) -> List[np.ndarray]:
        # convert image to grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # initialize list of cutouts
        results = []

        # extract cutout for each level (receptive field)
        for level in range(self.__num_of_levels):
            result = LineProcessor.__extract_cutout(image_gray, keyline, cutout_width * pow(2, level + 1),
                                                    cutout_height * 2, cutout_width, cutout_height)
            results.append(result)

        return results

    @staticmethod
    def __compute_border_width(cutout_width: int, cutout_height: int) -> int:
        return int(max(cutout_width, cutout_height))

    @staticmethod
    def __compute_gabor_kernels() -> List[np.ndarray]:
        # CV_PI isn't available in OpenCV for Python
        pi = np.float_(3.1415926535897932384626433832795)

        # compute thetas
        thetas = []
        angle_steps = 2
        angle_inc = pi / np.float_(angle_steps)
        for step in range(angle_steps):
            thetas.append(np.float_(angle_inc * np.float_(step) + 0.0))

        # compute gabor kernels
        gabor_kernels = []
        sigma = float(1)
        lambd = float(2 * sigma)
        gamma = float(1)
        psi = np.float_(pi * 0.5 * 0.0)
        for theta in thetas:
            k = max(1, int(5 * sigma))
            gabor_kernel = cv2.getGaborKernel((k, k), sigma, float(theta), lambd, gamma, float(psi), cv2.CV_32F)
            gabor_kernel = gabor_kernel / np.float32(6.28)  # scale with 2 * PI to keep result in range
            gabor_kernels.append(gabor_kernel)

        return gabor_kernels

    @staticmethod
    def __extract_cutout(image: np.ndarray, keyline: cv2.line_descriptor_KeyLine, cutout_width: int,
                         cutout_height: int, dest_width: int = None, dest_height: int = None) -> np.ndarray:
        # check if destination size is given
        dest_width = (cutout_width if dest_width is None else dest_width)
        dest_height = (cutout_height if dest_height is None else dest_height)

        # compute border width based on cutout size
        border_width = LineProcessor.__compute_border_width(cutout_width, cutout_height)

        # pad image with a black border
        image = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT)

        # create segment from keyline
        seg = Segment2D([keyline.startPointX, keyline.startPointY], [keyline.endPointX, keyline.endPointY])

        # create borders of cutout
        seg1 = seg.translated(seg.normal() * int(cutout_width / 2))
        seg2 = seg.translated(-1 * seg.normal() * int(cutout_width / 2))

        # adjust segments to padded image
        seg1.translate(border_width, border_width)
        seg2.translate(border_width, border_width)

        # cutout must have specified height
        real_height = np.float_(cutout_height)
        seg1 = seg1.resized(real_height)
        seg2 = seg2.resized(real_height)

        # get bounding rectangle of cutout
        rect_points = np.array([seg1.p1(), seg1.p2(), seg2.p1(), seg2.p2()], dtype=np.float32)
        rect_left, rect_top, rect_width, rect_height = cv2.boundingRect(rect_points)

        # extract ROI from image
        if len(image.shape) == 3:  # rgb images
            roi = image[rect_top:rect_top + rect_height, rect_left:rect_left + rect_width, :].copy()
        elif len(image.shape) == 2:  # grayscale images
            roi = image[rect_top:rect_top + rect_height, rect_left:rect_left + rect_width].copy()
        else:  # this _should_ never happen but let's satisfy the linter
            raise ValueError

        # source and destination points of cutout
        src_points = np.array([
            [seg1.x2() - rect_left, seg1.y2() - rect_top],
            [seg1.x1() - rect_left, seg1.y1() - rect_top],
            [seg2.x2() - rect_left, seg2.y2() - rect_top],
            [seg2.x1() - rect_left, seg2.y1() - rect_top]
        ], dtype=np.float32)
        dst_points = np.array([
            [0, 0],
            [0, dest_height],
            [dest_width, 0],
            [dest_width, dest_height]
        ], dtype=np.float32)

        # since we want to use affine transformation we need to specify exactly three points
        src_points = src_points[:3, :]
        dst_points = dst_points[:3, :]

        # calculate transformation matrix
        m = cv2.getAffineTransform(src_points, dst_points)

        # warp ROI into cutout using transformation matrix
        cutout = cv2.warpAffine(roi, m, (dest_width, dest_height))

        return cutout
