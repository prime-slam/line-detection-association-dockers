from enum import Enum
from typing import Union, Tuple

import cv2
import numpy as np

Point = Union[list, tuple, np.ndarray]


class Cache(Enum):
    MAGNITUDE = 'magnitude'
    DIRECTION = 'direction'
    UNIT = 'unit'
    CENTER = 'center'
    NORMAL = 'normal'
    ANGLE = 'angle'


class Segment2D:
    def p1(self) -> np.ndarray:
        return self.__p1

    def p2(self) -> np.ndarray:
        return self.__p2

    def x1(self) -> np.float_:
        return self.p1()[0]

    def y1(self) -> np.float_:
        return self.p1()[1]

    def x2(self) -> np.float_:
        return self.p2()[0]

    def y2(self) -> np.float_:
        return self.p2()[1]

    def magnitude(self) -> np.float_:
        is_cached, value = self.__is_cached(Cache.MAGNITUDE)

        if is_cached:
            return value
        else:
            magnitude = Segment2D.__magnitude_of_vector(self.direction())
            self.__cache[Cache.MAGNITUDE] = magnitude
            return magnitude

    def direction(self) -> np.ndarray:
        is_cached, value = self.__is_cached(Cache.DIRECTION)

        if is_cached:
            return value
        else:
            direction = np.array([self.x2() - self.x1(), self.y2() - self.y1()], dtype=np.float_)
            self.__cache[Cache.DIRECTION] = direction
            return direction

    def unit(self) -> np.ndarray:
        is_cached, value = self.__is_cached(Cache.UNIT)

        if is_cached:
            return value
        else:
            unit = self.direction() / self.magnitude()
            self.__cache[Cache.UNIT] = unit
            return unit

    def center(self) -> np.ndarray:
        is_cached, value = self.__is_cached(Cache.CENTER)

        if is_cached:
            return value
        else:
            center = np.array([(self.x2() + self.x1()) / 2, (self.y2() + self.y1()) / 2], dtype=np.float_)
            self.__cache[Cache.CENTER] = center
            return center

    def normal(self) -> np.ndarray:
        is_cached, value = self.__is_cached(Cache.NORMAL)

        if is_cached:
            return value
        else:
            dx = self.unit()[0]
            dy = self.unit()[1]
            normal = np.array([dy, -dx], dtype=np.float_)
            self.__cache[Cache.NORMAL] = normal
            return normal

    def angle(self) -> np.float_:
        is_cached, value = self.__is_cached(Cache.ANGLE)

        if is_cached:
            return value
        else:
            # https://github.com/opencv/opencv_contrib/blob/master/modules/line_descriptor/src/LSDDetector.cpp
            # x = np.array([self.x2() - self.x1()], dtype=np.float_)
            # y = np.array([self.y2() - self.y1()], dtype=np.float_)
            # angle = np.arctan2(y, x)
            x = self.normal()[1]
            y = -self.normal()[0]
            angle = np.arctan2(y, x)
            self.__cache[Cache.ANGLE] = angle
            return angle

    def to_keyline(self, class_id: int, img_max_size: int) -> cv2.line_descriptor_KeyLine:
        kl = cv2.line_descriptor_KeyLine()
        kl.startPointX = float(self.x1())
        kl.startPointY = float(self.y1())
        kl.endPointX = float(self.x2())
        kl.endPointY = float(self.y2())
        kl.sPointInOctaveX = kl.startPointX
        kl.sPointInOctaveY = kl.startPointY
        kl.ePointInOctaveX = kl.endPointX
        kl.ePointInOctaveY = kl.endPointY

        kl.angle = float(self.angle())
        kl.octave = int(0)
        kl.class_id = int(class_id)
        kl.lineLength = float(self.magnitude())
        kl.pt = (float(self.center()[0]), float(self.center()[1]))

        kl.response = float(self.magnitude() / img_max_size)
        kl.size = float(np.abs(self.x1() - self.x2()) * np.abs(self.y1() - self.y2()))
        kl.numOfPixels = int(np.floor(self.magnitude()) + 1)

        return kl

    def translated(self, vector: np.ndarray) -> 'Segment2D':
        dx = np.float_(vector[0])
        dy = np.float_(vector[1])
        p1 = np.array([dx + self.x1(), dy + self.y1()], dtype=np.float_)
        p2 = np.array([dx + self.x2(), dy + self.y2()], dtype=np.float_)
        return Segment2D(p1, p2)

    def translate(self, dx: float, dy: float) -> None:
        dx = np.float_(dx)
        dy = np.float_(dy)
        self.__remove_cached(Cache.CENTER)
        self.__p1 = np.array([self.x1() + dx, self.y1() + dy], dtype=np.float_)
        self.__p2 = np.array([self.x2() + dx, self.y2() + dy], dtype=np.float_)

    def resized(self, length: float, center: float = 0.5) -> 'Segment2D':
        length = np.float_(length)
        center = np.float_(center)

        if length < self.magnitude():
            new_center = self.p1() + self.direction() * center
            radius = length / 2.0 * self.unit()
            seg2d = Segment2D(new_center - radius, new_center + radius)
            seg2d.set_magnitude(length)
            return seg2d
        else:
            length_difference = length - self.magnitude()
            missing = length_difference / 2.0 * self.unit()
            seg2d = Segment2D(self.p1() - missing, self.p2() + missing)
            seg2d.set_magnitude(length)
            return seg2d

    def set_magnitude(self, length: float) -> None:
        length = np.float_(length)
        self.__remove_cached(Cache.MAGNITUDE)
        self.__cache[Cache.MAGNITUDE] = length

    def __init__(self, p1: Point, p2: Point, lazy: bool = True) -> None:
        self.__p1 = np.array([p1[0], p1[1]], np.float_)
        self.__p2 = np.array([p2[0], p2[1]], np.float_)
        self.__cache = {}

        if not lazy:
            self.magnitude()
            self.direction()
            self.unit()
            self.center()
            self.normal()
            self.angle()

    def __str__(self) -> str:
        return '[{p1}, {p2}]'.format(p1=str(self.p1()), p2=str(self.p2()))

    def __is_cached(self, key: Cache) -> Tuple[bool, any]:
        is_cached = key in self.__cache
        value = self.__cache.get(key, None)
        return is_cached, value

    def __remove_cached(self, key: Cache) -> any:
        return self.__cache.pop(key, None)

    @staticmethod
    def __magnitude_of_vector(vector: np.ndarray) -> np.float_:
        return np.sqrt(vector.dot(vector))
