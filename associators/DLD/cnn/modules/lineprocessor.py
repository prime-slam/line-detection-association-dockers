from typing import List

import cv2
import numpy as np

from .geometry import Segment2D

Keylines = List[cv2.line_descriptor_KeyLine]
Cutouts = List[np.ndarray]


class LineProcessor:
    def process(self, image: np.ndarray, keylines: Keylines) -> Cutouts:
        # get cutout parameters from current instance
        cutout_width = self.__cutout_width
        cutout_height = self.__cutout_height
        border_width = self.__border_width

        # pad image with a black border
        image = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT)

        # initialize list of cutouts
        cutouts = []

        # loop through all keylines
        for class_id in range(len(keylines)):
            # create segment from keyline
            kl = keylines[class_id]
            seg = Segment2D([kl.startPointX, kl.startPointY], [kl.endPointX, kl.endPointY])

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
            roi = image[rect_top:rect_top + rect_height, rect_left:rect_left + rect_width, :].copy()

            # source and destination points of cutout
            src_points = np.array([
                [seg1.x2() - rect_left, seg1.y2() - rect_top],
                [seg1.x1() - rect_left, seg1.y1() - rect_top],
                [seg2.x2() - rect_left, seg2.y2() - rect_top],
                [seg2.x1() - rect_left, seg2.y1() - rect_top]
            ], dtype=np.float32)
            dst_points = np.array([
                [0, 0],
                [0, real_height],
                [cutout_width, 0],
                [cutout_width, real_height]
            ], dtype=np.float32)

            # since we want to use affine transformation we need to specify exactly three points
            src_points = src_points[:3, :]
            dst_points = dst_points[:3, :]

            # calculate transformation matrix
            m = cv2.getAffineTransform(src_points, dst_points)

            # warp ROI into cutout using transformation matrix
            cutout = cv2.warpAffine(roi, m, (cutout_width, int(real_height)))

            # append new cutout to array
            cutouts.append(cutout)

        return cutouts

    def __init__(self, cutout_width: int = 27, cutout_height: int = 100) -> None:
        # set instance variables
        self.__cutout_width = int(cutout_width)
        self.__cutout_height = int(cutout_height)
        self.__border_width = int(max(self.__cutout_width, self.__cutout_height) / 2)
